from d3m import container
from d3m.primitive_interfaces import base, transformer
from d3m.metadata import base as metadata_base, hyperparams

from quickstart_primitives.config_files import config

__all__ = ('InputToOutputPrimitive',)

Inputs = container.DataFrame
Outputs = container.DataFrame


class Hyperparams(hyperparams.Hyperparams):
    """
    No hyper-parameters for this primitive.
    """
    pass


class InputToOutputPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    A sample primitive. The output dataframe is the input dataframe.
    """

    metadata = hyperparams.base.PrimitiveMetadata({
        "id": "30d5f2fa-4394-4e46-9857-2029ec9ed0e0",
        "version": config.VERSION,
        "name": "Passthrough primitive",
        "description": "A primitive which directly outputs the input.",
        "python_path": "d3m.primitives.operator.input_to_output.Quickstart",
        "primitive_family": metadata_base.PrimitiveFamily.OPERATOR,
        "algorithm_types": [metadata_base.PrimitiveAlgorithmType.IDENTITY_FUNCTION],
        "source": {
            "name": config.AUTHOR_NAME,
            "contact": config.AUTHOR_CONTACT,
            "uris": [config.REPOSITORY],
        },
        "keywords": ["passthrough"],
        "installation": [config.INSTALLATION],
    })

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        self.logger.warning('Hi, InputToOutputPrimitive.produce was called!')
        return base.CallResult(value=inputs)
