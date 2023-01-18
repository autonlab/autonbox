import unittest
import os

import pandas as pd
from statsforecast.arima import AutoARIMA

from d3m import runtime, index
from d3m.container import dataset
from d3m.metadata import problem
from d3m.metadata.base import ArgumentType, Context
from d3m.metadata.pipeline import Pipeline, PrimitiveStep

DATASET_LOCATION = "/home/mkowales/datasets/sunspots/"
PROBLEM_PATH = os.path.join(DATASET_LOCATION, "TRAIN", "problem_TRAIN", "problemDoc.json")
TRAIN_DOC_PATH = os.path.join(DATASET_LOCATION, "TRAIN", "dataset_TRAIN", "datasetDoc.json")
TEST_DOC_PATH = os.path.join(DATASET_LOCATION, "TEST", "dataset_TEST", "datasetDoc.json")
TRAIN_DATA_PATH = os.path.join(DATASET_LOCATION, "TRAIN", "dataset_TRAIN", "tables", "learningData.csv")
TEST_DATA_PATH = os.path.join(DATASET_LOCATION, "TEST", "dataset_TEST", "tables", "learningData.csv")

class AutoARIMAWrapperTestCase(unittest.TestCase):

    #hyperparams argument is a list of (name, data) tuples
    def construct_pipeline(self, hyperparams) -> Pipeline:
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

        # Step 5: autoARIMA
        step_5 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.time_series_forecasting.arima.AutonBox'))
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

    def run_pipeline(self, pipeline_description : Pipeline):
        # Loading problem description.
        problem_description = problem.get_problem(PROBLEM_PATH)

        # Loading train and test datasets.
        train_dataset = dataset.get_dataset(TRAIN_DOC_PATH)
        test_dataset = dataset.get_dataset(TEST_DOC_PATH)

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

        return test_predictions.sunspots
    
    '''
    def test_default_params(self):

        #run simple pipeline with AutoARIMA primitive
        pipeline_description = self.construct_pipeline(hyperparams=[])
        pipeline_predictions = self.run_pipeline(pipeline_description).to_numpy().flatten()

        #run statsforecast AutoARIMA directly
        train = pd.read_csv(TRAIN_DATA_PATH)
        test = pd.read_csv(TEST_DATA_PATH)
        y = train.sunspots.to_numpy().flatten()
        h = test.shape[0] #number of rows in test

        arima = AutoARIMA()
        arima.fit(y)
        direct_predictions = arima.predict(h=h, X=).to_numpy().flatten()

        print("direct:")
        print(direct_predictions)
        print("from pipeline:")
        print(pipeline_predictions)

        assert((direct_predictions == pipeline_predictions).all())
    '''

    def test_exogenous(self):

        #run simple pipeline with AutoARIMA primitive
        pipeline_description = self.construct_pipeline(hyperparams=[])
        pipeline_predictions = self.run_pipeline(pipeline_description).to_numpy().flatten()

        #run statsforecast AutoARIMA directly
        train = pd.read_csv(TRAIN_DATA_PATH)
        test = pd.read_csv(TEST_DATA_PATH)
        y = train.sunspots.to_numpy().flatten()
        X = train.loc[:, list(("sd", "observations"))].to_numpy()
        Xf = test.loc[:, list(("sd", "observations"))].to_numpy()
        h = test.shape[0] #number of rows in test

        arima = AutoARIMA()
        arima.fit(y, X=X)
        direct_predictions = arima.predict(h=h, X=Xf).to_numpy().flatten()

        print("direct:")
        print(direct_predictions)
        print("from pipeline:")
        print(pipeline_predictions)

        assert((direct_predictions == pipeline_predictions).all())

if __name__ == '__main__':
    unittest.main()