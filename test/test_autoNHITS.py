import unittest
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch
#from neuralforecast import NeuralForecast
from neuralforecast.tsdataset import TimeSeriesDataset
from neuralforecast.auto import AutoNHITS
from neuralforecast.losses.pytorch import MAE

from d3m import runtime, index
from d3m.container import dataset
from d3m.metadata import problem
from d3m.metadata.base import ArgumentType, Context
from d3m.metadata.pipeline import Pipeline, PrimitiveStep

class AutoNHITSTestCase(unittest.TestCase):

    #hyperparams argument is a list of (name, data) tuples
    def construct_pipeline(self, hyperparams) -> Pipeline:
        # Creating pipeline
        pipeline_description = Pipeline()
        pipeline_description.add_input(name='inputs')

        # Step 0: denormalize
        step_0 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.denormalize.Common'))
        step_0.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data='inputs.0')
        step_0.add_output('produce')
        pipeline_description.add_step(step_0)

        # Step 1: dataset_to_dataframe
        step_1 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.dataset_to_dataframe.Common'))
        step_1.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data='steps.0.produce')
        step_1.add_output('produce')
        pipeline_description.add_step(step_1)

        # Step 2: profiler
        # Automatically determine semantic types of columns
        step_2 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.schema_discovery.profiler.Common'))
        step_2.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data='steps.1.produce')
        step_2.add_output('produce')
        pipeline_description.add_step(step_2)

        # Step 3: column parser
        # automatically determine data types of colums (eg float, boolean, integer)
        step_3 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.column_parser.Common'))
        step_3.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data='steps.2.produce')
        step_3.add_output('produce')
        # not adding this hyperparameter messes it up
        # We have NO IDEA why (Merritt and Piggy)
        # Looking at the documentation for this primitive,
        # using defaults should be better.
        # if you use defaults it messes up the values in a lot of the columns
        step_3.add_hyperparameter(
            name='parse_semantic_types',
            argument_type=ArgumentType.VALUE,
            data=[
                "http://schema.org/Boolean",
                "http://schema.org/Integer",
                "http://schema.org/Float",
                "https://metadata.datadrivendiscovery.org/types/FloatVector"
                # "http://schema.org/DateTime" (adding this messes up the "year" column for some reason)
            ]
        )
        pipeline_description.add_step(step_3)

        # Step 4: extract_columns_by_semantic_types(targets)
        step_4 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common'))
        step_4.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data='steps.3.produce')
        step_4.add_output('produce')
        step_4.add_hyperparameter(
            name='semantic_types',
            argument_type=ArgumentType.VALUE,
            data=[
                "https://metadata.datadrivendiscovery.org/types/Target",
                "https://metadata.datadrivendiscovery.org/types/TrueTarget",
                "https://metadata.datadrivendiscovery.org/types/SuggestedTarget"
            ]
        )
        pipeline_description.add_step(step_4)

        # Step 5: extract_columns_by_semantic_types(attributes)
        step_5 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common'))
        step_5.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data='steps.3.produce')
        step_5.add_output('produce')
        step_5.add_hyperparameter(
            name='semantic_types',
            argument_type=ArgumentType.VALUE,
            data=['https://metadata.datadrivendiscovery.org/types/Attribute'],
        )
        pipeline_description.add_step(step_5)

        # Step 6: imputer
        # I think we only impute attributes but need to check this
        step_6 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_cleaning.imputer.SKlearn'))
        step_6.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data='steps.5.produce')
        step_6.add_output('produce')
        step_6.add_hyperparameter(
            name='use_semantic_types',
            argument_type=ArgumentType.VALUE,
            data=True
        )
        step_6.add_hyperparameter(
            name='return_result',
            argument_type=ArgumentType.VALUE,
            data='replace'
        )
        step_6.add_hyperparameter(
            name='strategy',
            argument_type=ArgumentType.VALUE,
            data='median'
        )
        step_6.add_hyperparameter(
            name='error_on_no_input',
            argument_type=ArgumentType.VALUE,
            data=False
        )
        pipeline_description.add_step(step_6)

        # Step 7: grouping field compose
        step_7 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.grouping_field_compose.Common'))
        step_7.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data='steps.6.produce')
        step_7.add_output('produce')
        pipeline_description.add_step(step_7)

        # Step 8: autoNHITS
        step_8 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.time_series_forecasting.nhits.AutonBox'))
        step_8.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data='steps.7.produce')
        step_8.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER, data='steps.4.produce')
        #add hyperparams from argument
        for h in hyperparams:
            (name, data) = h
            step_8.add_hyperparameter(
                name = name,
                argument_type = ArgumentType.VALUE,
                data = data
            )
        step_8.add_output('produce')
        pipeline_description.add_step(step_8)

        # Step 9: construct_predictions
        # This is a primitive which assures that the output of a standard pipeline has predictions
        # in the correct structure (e.g., there is also a d3mIndex column with index for every row).
        step_9 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.construct_predictions.Common'))
        step_9.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data='steps.8.produce')
        # This is a primitive which uses a non-standard second argument, named "reference".
        step_9.add_argument(name='reference', argument_type=ArgumentType.CONTAINER, data='steps.1.produce')
        step_9.add_output('produce')
        pipeline_description.add_step(step_9)

        # Final output
        pipeline_description.add_output(name='output predictions', data_reference='steps.9.produce')

        # print json for reference
        #print(pipeline_description.to_json())

        return pipeline_description
    
        '''
        # Creating pipeline
        pipeline_description = Pipeline()
        pipeline_description.add_input(name='inputs')

        # Step 0: dataset_to_dataframe
        step_0 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.dataset_to_dataframe.Common'))
        step_0.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data='inputs.0')
        step_0.add_output('produce')
        pipeline_description.add_step(step_0)

        # Step 1: profiler
        # Automatically determine semantic types of columns
        step_1 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.schema_discovery.profiler.Common'))
        step_1.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data='steps.0.produce')
        step_1.add_output('produce')
        pipeline_description.add_step(step_1)

        # Step 2: column parser
        step_2 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.column_parser.Common'))
        step_2.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data='steps.1.produce')
        step_2.add_output('produce')

        #not adding this hyperparameter messes it up
        #I have NO IDEA why
        #looking at the documentation for this primitive,
        #  using defaults should be better
        #  if if you use defaults it messes up the values in a lot of the columns
        step_2.add_hyperparameter(
            name='parse_semantic_types',
            argument_type=ArgumentType.VALUE,
            data=[
                "http://schema.org/Boolean",
                "http://schema.org/Integer",
                "http://schema.org/Float",
                "https://metadata.datadrivendiscovery.org/types/FloatVector"
            ]
        )

        pipeline_description.add_step(step_2)

        # Step 3: extract_columns_by_semantic_types(attributes)
        step_3 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common'))
        step_3.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data='steps.2.produce')
        step_3.add_output('produce')
        step_3.add_hyperparameter(
            name='semantic_types',
            argument_type=ArgumentType.VALUE,
            data=['https://metadata.datadrivendiscovery.org/types/Attribute'],
        )
        pipeline_description.add_step(step_3)

        # Step 4: extract_columns_by_semantic_types(targets)
        step_4 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common'))
        step_4.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data='steps.2.produce')
        step_4.add_output('produce')
        step_4.add_hyperparameter(
            name='semantic_types',
            argument_type=ArgumentType.VALUE,
            data=[
                "https://metadata.datadrivendiscovery.org/types/Target",
                "https://metadata.datadrivendiscovery.org/types/TrueTarget",
                "https://metadata.datadrivendiscovery.org/types/SuggestedTarget"
            ]
        )
        pipeline_description.add_step(step_4)

        # Step 5: autoNHITS
        step_5 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.time_series_forecasting.nhits.AutonBox'))
        step_5.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data='steps.3.produce')
        step_5.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER, data='steps.4.produce')
        #add hyperparams from argument
        for h in hyperparams:
            (name, data) = h
            step_5.add_hyperparameter(
                name = name,
                argument_type = ArgumentType.VALUE,
                data = data
            )
        step_5.add_output('produce')
        pipeline_description.add_step(step_5)

        # Step 6: construct_predictions
        # This is a primitive which assures that the output of a standard pipeline has predictions
        # in the correct structure (e.g., there is also a d3mIndex column with index for every row).
        step_6 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.construct_predictions.Common'))
        step_6.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data='steps.5.produce')
        # This is a primitive which uses a non-standard second argument, named "reference".
        step_6.add_argument(name='reference', argument_type=ArgumentType.CONTAINER, data='steps.0.produce')
        step_6.add_output('produce')
        pipeline_description.add_step(step_6)

        # Final output
        pipeline_description.add_output(name='output predictions', data_reference='steps.6.produce')

        # print json for reference
        #print(pipeline_description.to_json())

        return pipeline_description
        '''

    def run_pipeline(self, pipeline_description : Pipeline, dataset_location : str):
        problem_path = os.path.join(dataset_location, "TRAIN", "problem_TRAIN", "problemDoc.json")
        train_doc_path = os.path.join(dataset_location, "TRAIN", "dataset_TRAIN", "datasetDoc.json")
        test_doc_path = os.path.join(dataset_location, "TEST", "dataset_TEST", "datasetDoc.json")

        # Loading problem description.
        problem_description = problem.get_problem(problem_path)

        # Loading train and test datasets.
        train_dataset = dataset.get_dataset(train_doc_path)
        test_dataset = dataset.get_dataset(test_doc_path)

        print(train_dataset)
        print(test_dataset)

        # Fitting pipeline on train dataset.
        fitted_pipeline, train_predictions, fit_result = runtime.fit(
            pipeline_description,
            [train_dataset],
            problem_description=problem_description,
            context=Context.TESTING,
        )

        # Any errors from running the pipeline are captured and stored in
        # the result objects (together with any values produced until then and
        # pipeline run information). Here we just want to know if it succeed.
        fit_result.check_success()

        # Producing predictions using the fitted pipeline on test dataset.
        test_predictions, produce_result = runtime.produce(
            fitted_pipeline,
            [test_dataset],
        )
        produce_result.check_success()

        return test_predictions
    
    def run_direct(self, train, test, h):
        #run AutoNHITS directly
        future_exog = list(set(train.columns) - set(['ds', 'unique_id', 'y']))
        
        print("Fitting NeuralForecast AutoNHITS (direct)")
        print("train:")
        print(train)
        print("h:" + str(h))
        print("future exog: " + str(future_exog))

        nhits_config = {
            "input_size": 3*h,
            "n_pool_kernel_size": tune.choice(
                [3 * [1], 3 * [2], 3 * [4], [8, 4, 1], [16, 8, 1]]
            ),
            "n_freq_downsample": tune.choice(
                [
                    [168, 24, 1],
                    [24, 12, 1],
                    [180, 60, 1],
                    [60, 8, 1],
                    [40, 20, 1],
                    [1, 1, 1],
                ]
            ),
            "learning_rate": tune.loguniform(1e-4, 1e-1),
            "scaler_type" : 'robust',
            "max_steps": 100,  #TODO: change to 1000 after testing
            "batch_size": tune.choice([32, 64, 128, 256]),
            "windows_batch_size": tune.choice([128, 256, 512, 1024]),                                                                       # Initial Learning rate
            "random_seed": 1,
            "futr_exog_list" : future_exog
        }
        
        model = AutoNHITS(
                h=h,
                loss=MAE(),
                config=nhits_config,
                search_alg=HyperOptSearch(),
                num_samples=10)

        train, uids, last_dates, ds = TimeSeriesDataset.from_df(df=train)
        model.fit(train, val_size=h*2)

        y = test['y']
        del test['y']

        print("future:")
        print(test)
        dataset = TimeSeriesDataset.update_dataset(
            dataset=train, future_df=test
        )

        model.set_test_size(h)  # To predict h steps ahead
        model_fcsts = model.predict(dataset=dataset)
        #print("model_fcsts:")
        #print(model_fcsts)
        return(pd.DataFrame({"y": list(model_fcsts.flatten())}))
        #----------------------------------
        '''
        if issubclass(last_dates.dtype.type, np.integer):
            last_date_f = lambda x: np.arange(
            	x + 1, x + 1 + h, dtype=last_dates.dtype
        	)
        else:
            last_date_f = lambda x: pd.date_range(
                x + self.freq, periods=h, freq=self.freq
            )

        if len(np.unique(last_dates)) == 1:
            dates = np.tile(last_date_f(last_dates[0]), len(train))
        else:
            dates = np.hstack([last_date_f(last_date) for last_date in last_dates])
        
        idx = pd.Index(np.repeat(uids, h), name="unique_id")
        fcsts_df  = pd.DataFrame({"ds": dates}, index=idx)

        col_idx = 0
        fcsts = np.full((h * len(uids), 1), fill_value=np.nan)
        
        # Append predictions in memory placeholder
        output_length = len(model.loss.output_names)
        fcsts[:, col_idx : col_idx + output_length] = model_fcsts
        col_idx += output_length

        # Declare predictions pd.DataFrame
        fcsts = pd.DataFrame.from_records(fcsts, columns=cols, index=fcsts_df.index)
        fcsts_df = pd.concat([fcsts_df, fcsts], axis=1)
        '''
        #----------------------------------

    def test_nfsample(self):
        print("testing nf sample dataset")
        dataset_location = "/home/mkowales/datasets/nfsample/d3m"

        train_data_path = os.path.join(dataset_location, "TRAIN", "dataset_TRAIN", "tables", "learningData.csv")
        test_data_path = os.path.join(dataset_location, "TEST", "dataset_TEST", "tables", "learningData.csv")

        target_name = 'y'

        train = pd.read_csv(train_data_path)
        test = pd.read_csv(test_data_path)
        train['ds'] = pd.to_datetime(train['ds'])
        test['ds'] = pd.to_datetime(test['ds'])

        del train['d3mIndex']
        del test['d3mIndex']

        h = int(test.shape[0]/2)

        #----------

        #run AutoNHITS directly
        direct_predictions = self.run_direct(train, test, h)

        #run simple pipeline with AutoNHITS primitive
        pipeline_description = self.construct_pipeline(hyperparams=[])
        pipeline_predictions = self.run_pipeline(pipeline_description, dataset_location)
        pipeline_predictions = pipeline_predictions[target_name]

        print("direct:")
        print(direct_predictions)
        print(type(direct_predictions))
        print("from pipeline:")
        print(pipeline_predictions)
        print(type(pipeline_predictions))

        #predictions will not necessarily be identical but should be similar
        #assert((direct_predictions['y'] == pipeline_predictions).all())

    def test_sunspots(self):
        print("testing sunspots")
        
        dataset_location = "/home/mkowales/datasets/sunspots/d3m"

        train_data_path = os.path.join(dataset_location, "TRAIN", "dataset_TRAIN", "tables", "learningData.csv")
        test_data_path = os.path.join(dataset_location, "TEST", "dataset_TEST", "tables", "learningData.csv")

        target_name = 'sunspots'

        train = pd.read_csv(train_data_path)
        test = pd.read_csv(test_data_path)
        train['ds'] = pd.to_datetime(train['year'], format="%Y")
        test['ds'] = pd.to_datetime(test['year'], format="%Y")
        del train['year']
        del test['year']

        train['unique_id'] = ['a']*train.shape[0]
        test['unique_id'] = ['a']*test.shape[0]

        train.rename(columns={"sunspots":"y"}, inplace=True)
        test.rename(columns={"sunspots":"y"}, inplace=True)

        h = int(test.shape[0])

        # ----------
        # run AutoNHITS directly
        direct_predictions = self.run_direct(train, test, h)

        # run simple pipeline with AutoNHITS primitive
        pipeline_description = self.construct_pipeline(hyperparams=[])
        pipeline_predictions = self.run_pipeline(pipeline_description, dataset_location)
        pipeline_predictions = pipeline_predictions[target_name]

        print("direct:")
        print(direct_predictions)
        print("from pipeline:")
        print(pipeline_predictions)
        # predictions will not necessarily be identical but should be similar\
        # print("DEBUG: type(direct_predictions): %r", type(direct_predictions))
        # print("DEBUG: type(pipeline_predictions): %r", type(pipeline_predictions))

        rmse = self.ref_metric(direct_predictions, pipeline_predictions)
        epsilon = 30
        assert rmse < epsilon, "rmse: %f not < epsilon: %f" % (rmse, epsilon)

    def ref_metric(self, d1, d2):
        return np.sqrt(mean_squared_error(d1, d2))


if __name__ == '__main__':
    unittest.main()