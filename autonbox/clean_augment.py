import os
import pandas as pd

from d3m import container
from d3m import utils as d3m_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces.base import CallResult
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase

import autonbox

Inputs = container.Dataset
Outputs = container.Dataset

class Hyperparams(hyperparams.Hyperparams):
    original_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[str](''),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="List of the original columns"
    )
    percentage_missing = hyperparams.Uniform(
        lower=0.0, 
        upper=1.0, 
        default=0.5,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description='Percentage of missing data allowed (above this percentage and the line is deleted)'
    )


class CleanAugmentationPrimitive(TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
        Remove rows which haven't been augmented following the datamart augmentation 
        (any rows with more than percentage_missing columns which are resulting from augmentation)
        
        NB: This primitive results might reduce the number of row
    """

    metadata = metadata_base.PrimitiveMetadata({
        'id': 'fe0f1ac8-1d39-463a-b344-7bd498a31b92',
        'version': '0.1.0',
        'name': "Clean dataset of unaugmented rows",
        'python_path': 'd3m.primitives.data_cleaning.clean_augmentation.AutonBox',
        'source': {
            'name': autonbox.__author__,
            'uris': ['https://github.com/autonlab/autonbox'],
            'contact': 'mailto:vjeansel@andrew.cmu.edu'
        },
        "installation": [{
            "type": metadata_base.PrimitiveInstallationType.PIP,
            "package": "autonbox",
            "version": "0.2.4"
        }],
        'algorithm_types': [
            #metadata_base.PrimitiveAlgorithmType.ROW_SELECTION
            metadata_base.PrimitiveAlgorithmType.DATA_RETRIEVAL, #TODO: Delete when new algo released
        ],
        'primitive_family': metadata_base.PrimitiveFamily.DATA_CLEANING
    })

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        # Selection augmentation columns
        augmented_data = inputs['learningData'][[c for c in inputs['learningData'].columns if c not in self.hyperparams['original_columns']]]
        
        # Count absent data: at Dataset level: empty field
        absent = (augmented_data == '').mean(axis = 1)
        
        # Selection of the augmented lines
        output = inputs.copy()
        output['learningData'] = output['learningData'].loc[absent < self.hyperparams['percentage_missing']]

        return CallResult(output)
