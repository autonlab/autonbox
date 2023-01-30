import typing
from frozendict import FrozenOrderedDict

from d3m import container
from d3m.primitive_interfaces import base
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from d3m.metadata import base as metadata_base
from d3m.metadata import hyperparams, params
from d3m.exceptions import MissingValueError, PrimitiveNotFittedError

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
    fitted: bool
    new_training_data: bool
    autoARIMA: typing.Any

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
        #print("calling __init__")

        super().__init__(hyperparams=hyperparams)

        self._fitted = False
        self._training_target = None
        self._training_exogenous = None
        self._new_training_data = False
        self._autoARIMA = None

    def get_params(self) -> Params:
        #print("calling get_params")
        return Params(
            fitted = self._fitted,
            new_training_data = self._new_training_data,
            autoARIMA = self._autoARIMA
        )

    def set_params(self, *, params: Params) -> None:
        #print("calling set_params, params argument:")
        #print(params)
        self._fitted = params['fitted']
        self._new_training_data = params['new_training_data']
        self._autoARIMA = params['autoARIMA']

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        '''
        print("calling set_training_data")
        print("Inputs:")
        print(inputs)
        print("Outputs:")
        print(outputs)
        '''
        
        '''
        inputs is a dataframe that will be used as exogenous data, excepting time columns
        outputs is a dataframe containing one column, the time series that we want to predict future values of
        '''

        #TODO: check that outputs has one column
        #TODO: check that inputs and outputs have same number of rows
        #TODO: check at np.nan and np.inf are not present

        self._training_exogenous = inputs
        self._training_target = outputs
        self._new_training_data = True

    #private method
    #determine columns to be used as exogenous data from column semantic types
    def _format_exogenous(self, inputs):
        timestamp_cols = inputs.metadata.list_columns_with_semantic_types(
            (
                "https://metadata.datadrivendiscovery.org/types/Time",
            )
        )
        #print("timestamp cols: " + str(timestamp_cols))
        #TODO: raise error if there are multiple time cols or it is not a valid time series?

        grouping_cols = inputs.metadata.list_columns_with_semantic_types(
            (
                "https://metadata.datadrivendiscovery.org/types/GroupingKey",
                "https://metadata.datadrivendiscovery.org/types/SuggestedGroupingKey"
            )
        )
        #print("grouping cols: " + str(grouping_cols))
        #TODO: raise error if there are any grouping cols

        attribute_cols = inputs.metadata.list_columns_with_semantic_types(
            (
                "https://metadata.datadrivendiscovery.org/types/Attribute",
            )
        )
        #print("attribute cols: " + str(attribute_cols))
        
        exogenous_cols = list(set(attribute_cols) - set(grouping_cols + timestamp_cols))
        #print("exogneous_cols: " + str(exogenous_cols))

        #return None
        if (exogenous_cols == []):
            return None
        else:
            exogenous = inputs.iloc[:, exogenous_cols]
            #print("exogenous: ")
            #print(exogenous)
            X = exogenous.to_numpy().astype(float)
            return X 

    def fit(self, *, timeout: float = None, iterations: int = None) -> base.CallResult[None]:
        #print("Fitting StatsForecast AutoARIMA")

        #make hyperparams into local variables for convenience
        d = self.hyperparams['d']
        D = self.hyperparams['D']
        max_p = self.hyperparams['max_p']
        max_q = self.hyperparams['max_q']
        max_P = self.hyperparams['max_P']
        max_Q = self.hyperparams['max_Q']
        max_order = self.hyperparams['max_order']
        max_d = self.hyperparams['max_d']
        max_D = self.hyperparams['max_D']
        start_p = self.hyperparams['start_p']
        start_q = self.hyperparams['start_q']
        start_P = self.hyperparams['start_P']
        start_Q = self.hyperparams['start_Q']
        stationary = self.hyperparams['stationary']
        seasonal = self.hyperparams['seasonal']
        ic = self.hyperparams['ic']
        stepwise = self.hyperparams['stepwise']
        nmodels = self.hyperparams['nmodels']
        trace = self.hyperparams['trace']
        approximation = self.hyperparams['approximation']
        method = self.hyperparams['method']
        truncate = self.hyperparams['truncate']
        test = self.hyperparams['test']
        test_kwargs = self.hyperparams['test_kwargs']
        seasonal_test = self.hyperparams['seasonal_test']
        seasonal_test_kwargs = self.hyperparams['seasonal_test_kwargs']
        allowdrift = self.hyperparams['allowdrift']
        allowmean = self.hyperparams['allowmean']
        blambda = self.hyperparams['blambda']
        biasadj = self.hyperparams['biasadj']
        parallel = self.hyperparams['parallel']
        num_cores = self.hyperparams['num_cores']
        period = self.hyperparams['period']

        self._autoARIMA = AutoARIMA(
            d = d,
            D = D,
            max_p = max_p,
            max_q = max_q,
            max_P = max_P,
            max_Q = max_Q,
            max_order = max_order,
            max_d = max_d,
            max_D = max_D,
            start_p = start_p,
            start_q = start_q,
            start_P = start_P,
            start_Q = start_Q,
            stationary = stationary,
            seasonal = seasonal,
            ic = ic,
            stepwise = stepwise,
            nmodels = nmodels,
            trace = trace,
            approximation = approximation,
            method = method,
            truncate = truncate,
            test = test,
            test_kwargs = test_kwargs,
            seasonal_test = seasonal_test,
            seasonal_test_kwargs = seasonal_test_kwargs,
            allowdrift = allowdrift,
            allowmean = allowmean,
            blambda = blambda,
            biasadj = biasadj,
            parallel = parallel,
            num_cores = num_cores,
            period = period
        )

        if self._training_target is None:
            raise MissingValueError("fit() called before training data set, call set_training_data() first.")
        
        if self._fitted == True and self._new_training_data == False:
            self.logger.warning("Model is already fit and training data has not changed.  Model will be refit from scratch, but expect nothing to change.")

        #AutoARIMA takes a 1-dimensional ndarray for y
        y = self._training_target.to_numpy().flatten()
        #print("y")
        #print(y)

        #extract exogenous columns if there are any, and turn them into a 2d numpy array of floats.
        X = self._format_exogenous(self._training_exogenous)
        #print("X:")
        #print(X)

        self._autoARIMA.fit(y=y, X=X)
        self._fitted = True
        return base.CallResult(None)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        #print("calling produce")
        #print("Inputs:")
        #print(inputs)

        #inputs is non-target columns that can optionally be used as future exogenous data.

        if not self._fitted:
            raise PrimitiveNotFittedError("Primitive not fitted.")

        #predict for a number of periods corresponding to number of rows in inputs
        nrows = inputs.shape[0]
        #print("nrows:")
        #print(nrows)
        
        X = self._format_exogenous(inputs)
        #print("X:")
        #print(X)

        predictions = self._autoARIMA.predict(h=nrows, X=X, level=[])
        #print("predictions:")
        #print(predictions)
        output = container.DataFrame(predictions, generate_metadata=True)
        return base.CallResult(output)



