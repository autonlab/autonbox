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
    has_training_data: bool
    new_training_data: bool

    training_target: typing.Any
    training_attributes: typing.Any
    nf_train_data: typing.Any
    target_name: str
    exog_names: typing.List
    ngroups: int

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

        self._has_training_data = False
        self._new_training_data = False

        self._training_target = None
        self._training_attributes = None
        self._nf_train_data = None
        self._target_name = None
        self._exog_names = []
        self._ngroups = 0

        self._fitted = False
        self._nf = None

    def get_params(self) -> Params:
        #TODO: update
        print("calling get_params")
        return Params(
            has_training_data = self._has_training_data,
            new_training_data = self._new_training_data,
            training_target = self._training_target,
            training_attributes = self._training_attributes,
            nf_train_data = self._nf_train_data,
            target_name = self._target_name,
            exog_names = self._exog_names,
            ngroups = self._ngroups,
            fitted = self._fitted,
            nf = self._nf
        )

    def set_params(self, *, params: Params) -> None:
        #TODO: update
        print("calling set_params")
        self._has_training_data = params['has_training_data']
        self._new_training_data = params['new_training_data']

        self._training_target = params['training_target']
        self._training_attributes = params['training_attributes']
        self._nf_train_data = params['nf_train_data']
        self._target_name = params['target_name']
        self._exog_names = params['exog_names']
        self._ngroups = params['ngroups']

        self._fitted = params['fitted']
        self._nf = params['nf']

    #private method
    def _format_data(self, attributes, target=None):
        #transform data from d3m input format to neuralforcast ingest format
        print("formatting data for neuralforecast")

        #extract time column as series
        time_col_indices = attributes.metadata.list_columns_with_semantic_types(
            (
                "https://metadata.datadrivendiscovery.org/types/Time",
            )
        )
        print("timestamp cols: " + str(time_col_indices))
        #TODO: make sure there's only 1 timestamp col
        #TODO: make sure it's valid datetime
        print("time col before conversion to datetime:")
        time_col = attributes.iloc[:,time_col_indices[0]]
        print(time_col)
        time_col = pd.to_datetime(time_col)
        print("time col after conversion to datetime:")
        print(time_col)

        #extract grouping column as series
        group_col_indices = attributes.metadata.list_columns_with_semantic_types(
            (
                "https://metadata.datadrivendiscovery.org/types/GroupingKey",
                "https://metadata.datadrivendiscovery.org/types/SuggestedGroupingKey"
            )
        )
        print("grouping cols: " + str(group_col_indices))
        #TODO: make sure theres <=1 grouping col
        #TODO: make sure grouping col is valid
        if len(group_col_indices) > 0:
            group_col = attributes.iloc[:,group_col_indices[0]]
        else:
            #data has only 1 time series (no grouping col)
            #however neuralforecast still requires a grouping col
            #create a grouping col putting all rows in same group
            group_col = ['a']*attributes.shape[0]
        #TODO: ensure groups are all same length
        self._ngroups = len(set(group_col))

        #extract names of exogneous variable columns and save them
        attribute_col_indices = attributes.metadata.list_columns_with_semantic_types(
            (
                "https://metadata.datadrivendiscovery.org/types/Attribute",
            )
        )
        print("attribute cols: " + str(attribute_col_indices))
        exog_col_inidices = list(set(attribute_col_indices) - set(group_col_indices + time_col_indices))
        print("exogenous cols: " + str(exog_col_inidices))
        exogenous_colnames = [list(attributes.columns)[i] for i in exog_col_inidices]
        print("exogenous colnames: " + str(exogenous_colnames))
        self._exog_names = exogenous_colnames

        #construct dataframe formatted to be ingested by neuralforecast
        nf_df = attributes[exogenous_colnames]  #exogenous cols retain name from dataset
        #TODO: check that no exogenous cols are named "ds", "y" or "unique_id"
        nf_df['ds'] = time_col
        nf_df['unique_id'] = group_col

        #add target col if we're given one and save target colname
        if target is not None:
            target_colname = target.columns[0]
            nf_df['y'] = target[target_colname]
            self._target_name = target_colname

        return(nf_df)


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

        if self._has_training_data:
            self._new_training_data = True

        self._has_training_data = True

        #save data in d3m format
        self._training_attributes = inputs
        self._training_target = outputs

        #save data in neuralforecast format
        self._nf_train_data = self._format_data(inputs, outputs)
        #this method also sets self._target_name and self._exog_names

    def fit(self, *, timeout: float = None, iterations: int = None) -> base.CallResult[None]:
        #in order to fit NHITS, need to know forecasting horizon
        #so fit in the produce method and do nothing here
        print("calling fit, do nothing")
        return base.CallResult(None)

    #private method
    def _fit_nf(self, h):
        print("Fitting NeuralForecast AutoNHITS")

        print("train:")
        print(self._nf_train_data)

        nhits_config = {
            "max_steps": 100,                                                         # Number of SGD steps
            "learning_rate": tune.loguniform(1e-5, 1e-1),                             # Initial Learning rate
            "n_pool_kernel_size": tune.choice([[2, 2, 2], [16, 8, 1]]),               # MaxPool's Kernelsize
            "n_freq_downsample": tune.choice([[168, 24, 1], [24, 12, 1], [1, 1, 1]]), # Interpolation expressivity ratios                                            # Compute validation every 50 steps
            "random_seed": 1,
            "input_size": h*5,                                 # Size of input window
            "futr_exog_list" : self._exog_names,    # <- Future exogenous variables
            "scaler_type" : 'robust'
        }

        model = AutoNHITS(
                h=h,
                loss=MAE(),
                config=nhits_config,
                num_samples=10)
        
        inferred_freq = pd.infer_freq(self._nf_train_data['ds'])
        print("inferred freq: " + inferred_freq)
        self._nf = NeuralForecast(models=[model], freq=inferred_freq)

        self._nf.fit(df=self._nf_train_data, val_size=h*2)

    
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
            #contains one column with the target's name which is all 0's
            print("returning dummy data for in-sample predictions")
            nrows = inputs.shape[0]
            predictions = pd.DataFrame({self._target_name:[0]*nrows})

        else:
            #fit if we have not fit the model yet
            #refit if there is new training data
            if not self._fitted or self._new_training_data:
                #predict for a number of periods corresponding to number of rows in inputs
                h = int(inputs.shape[0]/self._ngroups)
                print("h:" + str(h))

                self._fit_nf(h)

                self._fitted = True
                self._new_training_data = False #we have fit on current train data, no longer new

            #TODO: check that self._nf not None
            future = self._format_data(inputs)
            print("future:")
            print(future)

            predictions = self._nf.predict(futr_df=future)
            print("raw predictions:")
            print(predictions)

            #grouping column will be returned as the index
            #change it to a normal column
            predictions.reset_index(inplace=True)
            
            #drop grouping and time columns, d3m doesnt like them in outputs
            predictions = predictions.drop(['unique_id', 'ds'], axis=1)

            #rename output column to name of original target column
            predictions.rename(
                columns={
                    'AutoNHITS':self._target_name
                },
                inplace=True
            )

            print("predictions to return:")
            print(predictions)

        #need to put predictions in right format for d3m
        output = container.DataFrame(predictions, generate_metadata=True)
        return base.CallResult(output)



