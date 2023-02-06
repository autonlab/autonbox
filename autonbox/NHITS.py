import typing
from frozendict import FrozenOrderedDict

from d3m import container
from d3m.primitive_interfaces import base
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from d3m.metadata import base as metadata_base
from d3m.metadata import hyperparams, params
from d3m.exceptions import MissingValueError, PrimitiveNotFittedError

import pandas as pd

from ray import tune
from neuralforecast.auto import AutoNHITS
from neuralforecast.losses.pytorch import MAE
from neuralforecast import NeuralForecast

import autonbox

"""
A wrapper primitive for AutoNHITS from NeuralForecast (https://nixtla.github.io/neuralforecast/models.html#autonhits)
More information on Neural Hierarchical Interpolation for Time Series (NHITS): https://nixtla.github.io/neuralforecast/models.nhits.html
For an intro to NeuralForecast, see https://nixtla.github.io/neuralforecast/examples/installation.html


TODO: Add more information here
"""

#not sure if necessary
#TODO: uncomment if this not being here causes errors
#__all__ = ('AutoNHITSPrimitive',)

Inputs = container.DataFrame
Outputs = container.DataFrame

class Params(params.Params):
    new_training_data: bool
    training_target: typing.Any
    training_attributes: typing.Any
    fitted: bool
    nf: typing.Any              #NeuralForecast object containing model

class Hyperparams(hyperparams.Hyperparams):
    pass

class AutoNHITSPrimitive(SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):

    metadata = metadata_base.PrimitiveMetadata({
        "id": "91c8bd09-cf10-4fde-a471-e092ef3df6b4",
        "version": "0.1.0",
        "name": "neuralforecast.models.AutoNHITS",
        "description": "Wrapper of the AutoNHITS model from the neuralforecast package",
        "python_path": "d3m.primitives.time_series_forecasting.nhits.AutonBox",
        "primitive_family": metadata_base.PrimitiveFamily.TIME_SERIES_FORECASTING,
        "algorithm_types": ["DEEP_NEURAL_NETWORK"],
        'source': {
            'name': autonbox.__author__,
            'uris': ['https://github.com/autonlab/autonbox'],
            'contact': 'mailto:mkowales@andrew.cmu.edu'
        },
        "keywords": ["time series", "forecasting", "deep neural network"],
        "installation": [{
            "type": metadata_base.PrimitiveInstallationType.PIP,
            "package": "autonbox",
            "version": autonbox.__version__
        }]
    })

    def __init__(self, *, hyperparams: Hyperparams) -> None:
        print("calling __init__")

        super().__init__(hyperparams=hyperparams)

        self._training_target = None
        self._training_attributes = None
        self._new_training_data = False
        self._fitted = False
        self._nf = None

    def get_params(self) -> Params:
        print("calling get_params")
        return Params(
            new_training_data = self._new_training_data,
            training_target = self._training_target,
            training_attributes = self._training_attributes,
            fitted = self._fitted,
            nf = self._nf
        )

    def set_params(self, *, params: Params) -> None:
        print("calling set_params")
        #print(params)
        self._new_training_data = params['new_training_data']
        self._training_target = params['training_target']
        self._training_attributes = params['training_attributes']
        self._fitted = params['fitted']
        self._nf = params['nf']

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        print("calling set_training_data")
        print("Inputs:")
        print(inputs)
        print("Outputs:")
        print(outputs)
        
        #inputs is a dataframe that will be used as exogenous data, excepting time columns
        #outputs is a dataframe containing one column, the time series that we want to predict future values of
        
        #TODO: check that outputs has one column
        #TODO: check that inputs and outputs have same number of rows
        #TODO: check at np.nan and np.inf are not present

        self._training_attributes = inputs
        self._training_target = outputs
        self._new_training_data = True

    #private method
    def _format_data(self, attributes, target=None):
        print("formatting data for neuralforecast")

        timestamp_cols = attributes.metadata.list_columns_with_semantic_types(
            (
                "https://metadata.datadrivendiscovery.org/types/Time",
            )
        )
        print("timestamp cols: " + str(timestamp_cols))
        #TODO: make sure there's only 1 timestamp col
        #TODO: make sure it's valid datetime
        time_colname = attributes.columns[timestamp_cols[0]]

        grouping_cols = attributes.metadata.list_columns_with_semantic_types(
            (
                "https://metadata.datadrivendiscovery.org/types/GroupingKey",
                "https://metadata.datadrivendiscovery.org/types/SuggestedGroupingKey"
            )
        )
        print("grouping cols: " + str(grouping_cols))
        #TODO: make sure theres <=1 grouping col
        #TODO: deal with no grouping cols scenario
        #TODO: make sure grouping col is valid
        grouping_colname = attributes.columns[grouping_cols[0]]

        attribute_cols = attributes.metadata.list_columns_with_semantic_types(
            (
                "https://metadata.datadrivendiscovery.org/types/Attribute",
            )
        )
        print("attribute cols: " + str(attribute_cols))
    
        exogenous_cols = list(set(attribute_cols) - set(grouping_cols + timestamp_cols))
        print("exogenous cols: " + str(exogenous_cols))
        exogenous_colnames = [list(attributes.columns)[i] for i in exogenous_cols]
        print("exogenous colnames: " + str(exogenous_colnames))

        nf_df = attributes.rename(columns={
            grouping_colname : 'unique_id',
            time_colname : 'ds'})

        if target is not None:
            target_colname = target.columns[0]
            nf_df['y'] = target[target_colname]
        else:
            target_colname = "n/a"

        nf_df['ds'] = pd.to_datetime(nf_df['ds'])

        return((nf_df, time_colname, grouping_colname, exogenous_colnames, target_colname))

    def fit(self, *, timeout: float = None, iterations: int = None) -> base.CallResult[None]:
        #in order to fit NHITS, need to know forecasting horizon
        #so fit in the produce method and do nothing here
        print("calling fit, do nothing")
        return base.CallResult(None)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        print("calling produce")
        print("Inputs:")
        print(inputs)
        #inputs is non-target columns that can optionally be used as future exogenous data.
        #also includes time and grouping columns

        if inputs.equals(self._training_attributes):
            #D3M likes to get in-sample predictions before actual forecasts
            #neuralforecast doesn't like to do that
            #so, if inputs match training data (i.e. D3M is looking for in-sample predictions)
            #return essentially dummy data
            #hopefully this will not mess anything up down the line
            #it doesn't seem like training predictions are really used despite D3M wanting them
            #and they dont really make sense for time series forecasting
            #dataframe that is the same length as expected output
            #contains one column called 'y' which is all 0's
            print("returning dummy data for in-sample predictions")
            nrows = inputs.shape[0]
            predictions = pd.DataFrame({'y':[0]*nrows})

        else:
            if not self._fitted:
                print("Fitting NeuralForecast AutoNHITS")

                #predict for a number of periods corresponding to number of rows in inputs
                h = int(inputs.shape[0]/2)
                print("h:")
                print(h)

                #turn training data into a format that neuralforecast likes
                (train, timename, groupname, exognames, targetname) = self._format_data(
                    self._training_attributes,
                    self._training_target)
                print("train:")
                print(train)

                nhits_config = {
                    "max_steps": 100,                                                         # Number of SGD steps
                    "learning_rate": tune.loguniform(1e-5, 1e-1),                             # Initial Learning rate
                    "n_pool_kernel_size": tune.choice([[2, 2, 2], [16, 8, 1]]),               # MaxPool's Kernelsize
                    "n_freq_downsample": tune.choice([[168, 24, 1], [24, 12, 1], [1, 1, 1]]), # Interpolation expressivity ratios                                            # Compute validation every 50 steps
                    "random_seed": 1,
                    "input_size": h*5,                                 # Size of input window
                    "futr_exog_list" : exognames,    # <- Future exogenous variables
                    "scaler_type" : 'robust'
                }

                model = AutoNHITS(
                        h=h,
                        loss=MAE(),
                        config=nhits_config,
                        num_samples=10)

                self._nf = NeuralForecast(models=[model], freq='M')

                self._nf.fit(df=train, val_size=h*2)
                self._fitted = True
                true_targetname = targetname

            #TODO: check that self._nf not None
            (future, timename, groupname, exognames, targetname) = self._format_data(inputs)
            print("future:")
            print(future)

            predictions = self._nf.predict(futr_df=future)

            #grouping column will be returned as the index
            #change it to a normal column named index
            predictions.reset_index(inplace=True)
            print("predictions:")
            print(predictions)

            predictions = predictions.drop(['unique_id', 'ds'], axis=1)
            #change column names back to what they were originally
            predictions.rename(
                columns={
                    #'index':groupname,
                    #'ds':timename,
                    'AutoNHITS':true_targetname
                },
                inplace=True
            )

            print("predictions:")
            print(predictions)

        #need to put predictions in right format for d3m
        output = container.DataFrame(predictions, generate_metadata=True)
        return base.CallResult(output)



