import os

from d3m import container
from d3m import utils as d3m_utils
from d3m.primitive_interfaces import base
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from d3m.metadata import base as metadata_base
from d3m.metadata import hyperparams

import autonbox
from statsforecast.arima import AutoARIMA

__all__ = ('AutoARIMAWrapperPrimitive',)

Inputs = container.DataFrame
Outputs = container.DataFrame

class Hyperparams(hyperparams.Hyperparams):
    h = hyperparams.UniformInt(
        lower=1,
        upper=100000,
        default=100,
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter"],
        description="Number of periods for forecasting"
    )


class AutoARIMAWrapperPrimitive(SupervisedLearnerPrimitiveBase[Inputs, Outputs, Hyperparams]):

    """
    #TODO: add docstring
    """
    metadata = hyperparams.base.PrimitiveMetadata({
        "id": "434d4d25-dd61-4a32-a624-0f983995e189",
        "version": "0.1.0",
        "name": "AutoARIMA Wrapper",
        "description": "Wrapper of the AutoARIMA class from statsforecast package",
        "python_path": "d3m.primitives.time_series_forecasting.arima.AutonBox",
        "primitive_family": metadata_base.PrimitiveFamily.TIME_SERIES_FORECASTING,
        "algorithm_types": [], #TODO add these
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

    def fit(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[None]:
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
        self.autoArima = AutoARIMA() #init with params from self.hyperparams
        self.autoArima.fit(y=inputs) #add exogenous variables
        #set is_fit to true
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
        predictions = self.autoArima.predict(h=self.hyperparams['h'])
        return base.CallResult(value=inputs)