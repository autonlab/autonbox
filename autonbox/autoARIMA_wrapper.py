import os
import typing
from frozendict import FrozenOrderedDict

from d3m import container
from d3m import utils as d3m_utils
from d3m.primitive_interfaces import base
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from d3m.metadata import base as metadata_base
from d3m.metadata import hyperparams, params
from d3m.exceptions import MissingValueError, PrimitiveNotFittedError

import autonbox
from statsforecast.arima import AutoARIMA

__all__ = ('AutoARIMAWrapperPrimitive',)

Inputs = container.DataFrame
Outputs = container.DataFrame

class Params(params.Params):
    pass

class Hyperparams(hyperparams.Hyperparams):

    h = hyperparams.UniformInt(     #TODO: maybe change this to Bounded?
        lower=1,
        upper=1000,
        default=20,
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter"],
        description="Number of periods for forecasting"
    )

    '''
    X_fit = hyperparams.Union(
        configuration=FrozenOrderedDict([
            ("none",
                hyperparams.Constant(
                    default=None
                )
            ),
            ("X", 
                hyperparams.Hyperparameter(
                    default=[[0,0],[0,0]]
                )
            )
        ]),
        default="none",
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter"],
        description="An optional 2-d array of exogenous variables (float) to be passed to AutoARIMA.fit()"
    )

    X_predict = hyperparams.Union(
        configuration=FrozenOrderedDict([
            ("none",
                hyperparams.Constant(
                    default=None
                )
            ),
            ("X", 
                hyperparams.Hyperparameter(
                    default=[[0,0],[0,0]]
                )
            )
        ]),
        default="none",
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter"],
        description="An optional 2-d array of future exogenous variables (float) to be passed to AutoARIMA.predict()."
    )
    #TODO: X_predict Should have dimensions of (self.h, len(self.X_fit[0]). Is there a way to ensure this is the case?
    '''

    level = hyperparams.Union(
        configuration=FrozenOrderedDict([
            ("none",
                hyperparams.Constant(
                    default=None
                )
            ),
            ("level", 
                hyperparams.Hyperparameter(
                    default=(0, 0)
                )
            )
        ]),
        default="none",
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter"],
        description="An optional tuple of ints representing confidence levels for prediction intervals"
    )

    d = hyperparams.Union(
        configuration=FrozenOrderedDict([
            ("auto",
                hyperparams.Constant(
                    default=None
                )
            ),
            ("manual", 
                hyperparams.UniformInt(
                    lower=1,
                    upper=10,
                    default=2
                )
            )
        ]),
        default="auto",
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter"],
        description="Order of first-differencing.  Either set manually, or have it be chosen automatically."
    )
    #TODO: i think test only matters if d is None?  update hyperparams to reflect this

    D = hyperparams.Union(
        configuration=FrozenOrderedDict([
            ("auto",
                hyperparams.Constant(
                    default=None
                )
            ),
            ("manual", 
                hyperparams.UniformInt(
                    lower=1,
                    upper=10,
                    default=2
                )
            )
        ]),
        default="auto",
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter"],
        description="Order of seasonal-differencing.  Either set manually, or have it be chosen automatically."
    )
    #TODO: I think seasontest only matters if D is none?  update hyperparams to reflect this

    max_p = hyperparams.UniformInt(
        lower=1,
        upper=100,
        default=5,
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter"],
        description="Maximum value of p"
    )

    max_q = hyperparams.UniformInt(
        lower=1,
        upper=100,
        default=5,
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter"],
        description="Maximum value of q"
    )

    max_P = hyperparams.UniformInt(
        lower=1,
        upper=100,
        default=2,
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter"],
        description="Maximum value of P"
    )

    max_Q = hyperparams.UniformInt(
        lower=1,
        upper=100,
        default=2,
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter"],
        description="Maximum value of Q"
    )

    max_order = hyperparams.UniformInt(
        lower=1,
        upper=100,
        default=5,
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter"],
        description="Maximum value of p+q+P+Q if model selection is not stepwise"
    )

    max_d = hyperparams.UniformInt(
        lower=1,
        upper=10,
        default=2,
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter"],
        description="Maximum number of non-seasonal differences"
    )

    max_D = hyperparams.UniformInt(
        lower=1,
        upper=10,
        default=1,
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter"],
        description="Maximum number of seasonal differences"
    )

    start_p = hyperparams.UniformInt(
        lower=1,
        upper=10,
        default=2,
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter"],
        description="Starting value of p in stepwise procedure"
    )

    start_q = hyperparams.UniformInt(
        lower=1,
        upper=10,
        default=2,
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter"],
        description="Starting value of q in stepwise procedure"
    )

    start_P = hyperparams.UniformInt(
        lower=1,
        upper=10,
        default=1,
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter"],
        description="Starting value of P in stepwise procedure"
    )

    start_Q = hyperparams.UniformInt(
        lower=1,
        upper=10,
        default=1,
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter"],
        description="Starting value of Q in stepwise procedure"
    )

    stationary = hyperparams.UniformBool(
        default=False,
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter"],
        description="If False, restricts search to non-seasonal models."
    )

    ic = hyperparams.Enumeration(
        values=["aic", "aicc", "bic"],
        default="aicc",
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter"],
        description="information criterion used in model selection"
    )

    stepwise = hyperparams.UniformBool(
        default=True,
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter"],
        description="If True, will do stepwise selection (faster).  Otherwise, it searches over all models.  Non-stepwise selection can be very slow, especially for seasonal models."
    )

    nmodels=hyperparams.UniformInt(
        lower=1,
        upper=500,
        default=94,
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter"],
        description="Maximum number of models considered in the stepwise search."
    )

    trace=hyperparams.UniformBool(
        default=False,
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter"],
        description="If True, the list of ARIMA models considered will be reported."
    )

    approximation = hyperparams.Enumeration(
        values = [True, False, None],
        default=None,
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter"],
        description="If True, estimation is via conditional sums of squares and the information criteria used for model selection are approximated. The final model is still computed using maximum likelihood estimation. Approximation should be used for long time series or a high seasonal period to avoid excessive computation times.  If set to None, AutoARIMA will decide whether to approximate."
    )

    method = hyperparams.Enumeration(
        values = ("CSS", "CSS-ML", "ML", None),
        default=None,
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter"],
        description="fitting method: maximum likelihood or minimize conditional sum-of-squares.  If None, will choose automatically"
    )

    truncate = hyperparams.Union(
        configuration=FrozenOrderedDict([
            ("none",
                hyperparams.Constant(
                    default=None
                )
            ),
            ("truncate", 
                hyperparams.UniformInt(
                    lower=1,
                    upper=100,
                    default=50
                )
            )
        ]),
        default="none",
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter"],
        description="An integer value indicating how many observations to use in model selection.  The last truncate values of the series are used to select a model when truncate is not None and approximation=True. All observations are used if either truncate=None or approximation=False."
    )

    test = hyperparams.Enumeration(
        values=("kpss",),
        default="kpss",
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter"],
        description="Type of unit root test to use. See ndiffs for details."
    )

    test_kwargs = hyperparams.Constant(
        default = {},
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter"],
        description = "A dictionary of keyword arguments to be passed to the unit root test.  At the time of writing, the only argument that is read is alpha."
    )

    seasonal_test = hyperparams.Enumeration(
        values=("seas", "ocsb"),
        default="seas",
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter"],
        description="This determines which method is used to select the number of seasonal differences. The default method is to use a measure of seasonal strength computed from an STL decomposition. Other possibilities involve seasonal unit root tests."
    )

    seasonal_test_kwargs = hyperparams.Constant(
        default = {},
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter"],
        description = "A dictionary of keyword arguments to be passed to the seasonal unit root test.  At the time of writing, the only argument that is read is alpha."
    )

    allowdrift = hyperparams.UniformBool(
        default=True,
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter"],
        description="If True, models with drift terms are considered."
    )

    allowmean = hyperparams.UniformBool(
        default=True,
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter"],
        description="If True, models with a non-zero mean are considered."
    )

    blambda = hyperparams.Union(
        configuration=FrozenOrderedDict([
            ("none or auto",
                hyperparams.Enumeration(
                    values = (None, "auto"),
                    default=None
                )
            ),
            ("blambda", 
                hyperparams.Uniform(
                    lower=-5.0,
                    upper=5.0,
                    default=0.0
                )
            )
        ]),
        default="none or auto",
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter"],
        description="Box-Cox transformation parameter.  If lambda=\"auto\", then a transformation is automatically selected using BoxCox.lambda. The transformation is ignored if None. Otherwise, data transformed before model is estimated."
    )

    biasadj = hyperparams.UniformBool(
        default=False,
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter"],
        description="Use adjusted back-transformed mean for Box-Cox transformations.  If transformed data is used to produce forecasts and fitted values, a regular back transformation will result in median forecasts. If biasadj is True, an adjustment will be made to produce mean forecasts and fitted values."
    )

    parallel = hyperparams.UniformBool(
        default=False,
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ResourcesUseParameter"],
        description="If True and stepwise = False, then the specification search is done in parallel. This can give a significant speedup on multicore machines."
    )

    num_cores = hyperparams.Union(
        configuration=FrozenOrderedDict([
            ("none",
                hyperparams.Constant(
                    default=None
                )
            ),
            ("num_cores", 
                hyperparams.UniformInt(
                    lower=1,
                    upper=20,
                    default=2
                )
            )
        ]),
        default="num_cores",
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ResourcesUseParameter"],
        description="Allows the user to specify the amount of parallel processes to be used if parallel = True and stepwise = False. If None, then the number of logical cores is automatically detected and all available cores are used."
    )

    period = hyperparams.UniformInt(
        lower=1,
        upper=1000,
        default=1,
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter"],
        description="Number of observations per unit of time. For example 24 for Hourly data."
    )


class AutoARIMAWrapperPrimitive(SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):

    """
    #TODO: add docstring
    """
    metadata = metadata_base.PrimitiveMetadata({
        "id": "434d4d25-dd61-4a32-a624-0f983995e189",
        "version": "0.1.0",
        "name": "AutoARIMA Wrapper",
        "description": "Wrapper of the AutoARIMA class from statsforecast package",
        "python_path": "d3m.primitives.time_series_forecasting.arima.AutonBox",
        "primitive_family": metadata_base.PrimitiveFamily.TIME_SERIES_FORECASTING,
        "algorithm_types": ["AUTOREGRESSIVE_INTEGRATED_MOVING_AVERAGE"],
        'source': {
            'name': autonbox.__author__,
            'uris': ['https://github.com/autonlab/autonbox'],
            'contact': 'mailto:mkowales@andrew.cmu.edu'
        },
        "keywords": ["ARIMA", "time series", "forecasting"],
        "installation": [{
            'type': metadata_base.PrimitiveInstallationType.PIP,

            'package_uri': 'git+https://github.com/autonlab/autonbox.git@{git_commit}#egg=autonbox'.format(
                git_commit=d3m_utils.current_git_commit(os.path.dirname(__file__)),
            ),
        }]
    })

    def __init__(self, *, hyperparams: Hyperparams) -> None:
        super().__init__(hyperparams=hyperparams)

        self._fitted = False
        self._training_inputs = None
        self._training_outputs = None
        self._new_training_data = False
        self._autoArima = AutoARIMA() #TODO: init with params from self.hyperparams

    def get_params(self) -> Params:
        return self._params

    def set_params(self, *, params: Params) -> None:
        self._params = params

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        
        #print("Inputs:")
        #print(inputs)
        #print("Outputs:")
        #print(outputs)
        
        #inputs is optional exogenous data (can be None)
        #TODO: add a hyperparam for is_exogenous?  or exogenous_cols?

        #outputs is the time series that we want to predict future values of
        #TODO: check that outputs has one column

        self._training_exogenous = inputs
        self._training_target = outputs
        self._new_training_data = True

    def fit(self, *, timeout: float = None, iterations: int = None) -> base.CallResult[None]:
        """From Kin's code:
        Fit the AutoARIMA estimator
        Fit an AutoARIMA to a time series (numpy array) `y`
        and optionally exogenous variables (numpy array) `X`.
        Parameters
        ----------
        y: array-like of shape (n,)
            One-dimensional numpy array of floats without `np.nan` or `np.inf`
            values.
        X: array-like of shape (n, n_x) optional (default=None)
            An optional 2-d numpy array of exogenous variables (float).
        """
        if self._training_target is None:
            raise MissingValueError("fit() called before training data set, call set_training_data() first.")
        
        if self._fitted == True and self._new_training_data == False:
            self.logger.warning("Model is already fit and training data has not changed.  Model will be refit from scratch, but expect nothing to change.")

        #print("training inputs:")
        #print(self._training_inputs)

        #Kin's code takes a 1-dimensional ndarray for y
        y = self._training_target.to_numpy().flatten()
        #print("inputs to kin's code:")
        #print(y)

        self._autoArima.fit(y=y, X=self._training_exogenous)
        self._fitted = True
        return base.CallResult[None]

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        """From Kin's code:
        Forecast future values using a fitted AutoArima.
        Parameters
        ----------
        h: int
            Number of periods for forecasting.
        X: array-like of shape (h, n_x) optional (default=None)
            Future exogenous variables.
        level: int
            Confidence level for prediction intervals.
        Returns
        -------
        forecasts : pandas dataframe of shape (n, 1 + len(level))
            The array of fitted values.
            The confidence intervals for the forecasts are returned if
            level is not None.
        """

        #inputs is optional future exogenous data (can be None)
        #TODO: check that future and past exogenous data match

        if not self._fitted:
            raise PrimitiveNotFittedError("Primitive not fitted.")

        predictions = self._autoArima.predict(h=self.hyperparams['h'], X=inputs, level=self.hyperparams['level'])
        return base.CallResult(value=predictions)



