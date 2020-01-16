import os
import typing
import numpy as np
import pandas as pd

from d3m import container
from d3m import utils as d3m_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces.base import CallResult
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
from d3m.primitives.data_preprocessing.horizontal_concat import DSBOX

import autonbox

Inputs = container.List
Outputs = container.DataFrame

class MergePartialPredictionsPrimitive(TransformerPrimitiveBase[Inputs, Outputs, hyperparams.Hyperparams]):
    """
        Merge predictions of multiple models
        Useful if model do not produce predictions for each points and that it is necessary
        to merge those predictions (the first non nan will be returned)
    """

    metadata = metadata_base.PrimitiveMetadata({
        'id': '1cc95f70-0716-11ea-9762-3dd2bb86dde8',
        'version': '0.1',
        'name': "Merge predictions of multiple models",
        'python_path': 'd3m.primitives.data_transformation.merge_partial_predictions.AutonBox',
        'source': {
            'name': "Vincent Jeanselme",
            'uris': ['https://github.com/autonlab/autonbox'],
            'contact': 'mailto:vjeansel@andrew.cmu.edu'
        },
        'installation': [{
            'type': metadata_base.PrimitiveInstallationType.PIP,

            'package_uri': 'git+https://github.com/autonlab/autonbox.git@{git_commit}#egg=autonbox'.format(
                git_commit=d3m_utils.current_git_commit(os.path.dirname(__file__)),
            ),
        }],
        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.DATA_CONVERSION, #TODO: Choose
        ],
        'primitive_family': metadata_base.PrimitiveFamily.DATA_TRANSFORMATION
    })

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        # Merge inputs
        output = pd.concat(inputs, axis = 1)
        output.metadata = inputs[-1].metadata
        
        # Propagate best non nan score
        output = output.T.fillna(method = 'bfill').T.iloc[:, :1]
        
        return CallResult(output)