import os

from d3m import container
from d3m import utils as d3m_utils
from d3m.primitive_interfaces import base
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from d3m.metadata import base as metadata_base, hyperparams

import autonbox

__all__ = ('AutoARIMAWrapperPrimitive',)

Inputs = container.DataFrame
Outputs = container.DataFrame

class Hyperparams(hyperparams.Hyperparams):
    """
    No hyper-parameters for this primitive.
    """
    pass


class AutoARIMAWrapperPrimitive(SupervisedLearnerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    A sample primitive. The output dataframe is the input dataframe.
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

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        self.logger.warning('Hi, InputToOutputPrimitive.produce was called!')
        return base.CallResult(value=inputs)
