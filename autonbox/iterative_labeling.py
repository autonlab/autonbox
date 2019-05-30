import os

from d3m import container, utils as d3m_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.metadata import params
from d3m.metadata.base import PrimitiveFamily
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces import base
from d3m.primitives.classification.random_forest import SKlearn as SKRandomForestClassifier

import autonbox

Input = container.DataFrame
Output = container.DataFrame


class IterativeLabelingParams(params.Params):
    is_fitted: bool


class IterativeLabelingHyperparams(hyperparams.Hyperparams):
    blackbox = hyperparams.Primitive[SupervisedLearnerPrimitiveBase](
        primitive_families=[PrimitiveFamily.CLASSIFICATION],
        default=SKRandomForestClassifier,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description='Black box model for the classification.')


class IterativeLabelingPrimitive(SupervisedLearnerPrimitiveBase[Input, Output, IterativeLabelingParams,
                                                                IterativeLabelingHyperparams]):
    """
    Blackbox based iterative labeling for semi-supervised classification
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '6bb5824f-cf16-4615-8643-8c1758bd6751',
            'version': '0.1.0',
            "name": "Iterative labeling for semi-supervised learning",
            'description': "Blackbox based iterative labeling for semi-supervised classification",
            'python_path': 'd3m.primitives.semisupervised_classification.bbil.AutonBox',
            'source': {
                'name': autonbox.__author__,
                'uris': ['https://github.com/autonlab/autonbox'],
                'contact': 'mailto:donghanw@cs.cmu.edu'
            },
            'installation': [{
                'type': metadata_base.PrimitiveInstallationType.PIP,

                'package_uri': 'git+https://github.com/autonlab/autonbox.git@{git_commit}#egg=autonbox'.format(
                    git_commit=d3m_utils.current_git_commit(os.path.dirname(__file__)),
                ),
            }],
            'algorithm_types': [
                # FIXME consider adding type in algorithm_types @see
                # https://gitlab.com/datadrivendiscovery/d3m/blob/devel/d3m/metadata/schemas/v0/definitions.json#L1945
                metadata_base.PrimitiveAlgorithmType.BINARY_CLASSIFICATION,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.SEMISUPERVISED_CLASSIFICATION,
        },
    )

    def __init__(self, *, hyperparams: IterativeLabelingHyperparams) -> None:
        super().__init__(hyperparams=hyperparams)
        self._clf = None
        self._is_fitted = False
        self.X = None
        self.Y = None

    def __getstate__(self):
        return (
            self.hyperparams, self._clf, self._is_fitted)

    def __setstate__(self, state):
        self.hyperparams, self._clf, self._is_fitted = None

    def fit(self, *, timeout: float = None, iterations: int = None) -> base.CallResult[None]:
        primitive = self.hyperparams['blackbox']

        X = self.X
        y = self.y

        labeledIx = y[y.notnull().any(axis=1)].index
        unlabeledIx = y[y.isnull().any(axis=1)].index

        labeledX = X.iloc[labeledIx]
        labeledy = y.iloc[labeledIx]

        self._clf.fit(labeledX, labeledy)
        predictions = self._clf.predict(X.iloc[unlabeledIx])

        for i in range(len(predictions)):
            row = unlabeledIx[i]
            y.iat[row, 0] = predictions[i]

        self._clf.fit(X, y)
        self._is_fitted = True

        return base.CallResult(None)

    def set_training_data(self, *, inputs: Input, outputs: Output) -> None:
        """
         Sets input and output feature space.

        :param inputs:
        :param outputs:
        :return:
        """
        self.X = inputs
        self.y = outputs

    def get_params(self) -> IterativeLabelingParams:
        return IterativeLabelingParams(is_fitted=self._is_fitted)

    def set_params(self, *, params: IterativeLabelingParams) -> None:
        self._is_fitted = params['is_fitted']

    def produce(self, *, inputs: Input, timeout: float = None, iterations: int = None) -> base.CallResult[Output]:
        output = self._clf.predict(inputs)
        return base.CallResult(output)
