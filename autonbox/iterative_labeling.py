import os
import numpy as np

import d3m.metadata
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
    iters = hyperparams.UniformInt(lower=1, upper=100, default=5,
                                   semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
                                   description='The number of iterations of labeling')
    frac = hyperparams.Uniform(lower=0.01, upper=1.0, default=0.2,
                                 semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
                                 description='The fraction of unlabeled item to label')
    blackbox = hyperparams.Primitive[SupervisedLearnerPrimitiveBase](
        primitive_families=[PrimitiveFamily.CLASSIFICATION],
        default=SKRandomForestClassifier,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description='Black box model for the classification')


class IterativeLabelingPrimitive(SupervisedLearnerPrimitiveBase[Input, Output, IterativeLabelingParams,
                                                                IterativeLabelingHyperparams]):
    """
    Blackbox based iterative labeling for semi-supervised classification
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '6bb5824f-cf16-4615-8643-8c1758bd6751',
            'version': '0.2.1',
            "name": "Iterative labeling for semi-supervised learning",
            'description': "Blackbox based iterative labeling for semi-supervised classification",
            'python_path': 'd3m.primitives.semisupervised_classification.iterative_labeling.AutonBox',
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
                metadata_base.PrimitiveAlgorithmType.ITERATIVE_LABELING,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.SEMISUPERVISED_CLASSIFICATION,
        },
    )

    def __init__(self, *, hyperparams: IterativeLabelingHyperparams) -> None:
        super().__init__(hyperparams=hyperparams)
        self._prim_instance = None
        self._is_fitted = False
        self.X = None
        self.y = None
        self._iters = hyperparams['iters']
        self._frac = hyperparams['frac']

    def __getstate__(self):
        return (
            self.hyperparams, self._prim_instance, self._is_fitted)

    def __setstate__(self, state):
        self.hyperparams, self._prim_instance, self._is_fitted = state

    def fit(self, *, timeout: float = None, iterations: int = None) -> base.CallResult[None]:
        X = self.X
        y = self.y

        primitive = self.hyperparams['blackbox']
        primitive_hyperparams = primitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        custom_hyperparams = {'n_estimators': 100}
        if isinstance(primitive, d3m.primitive_interfaces.base.PrimitiveBaseMeta):  # is a class
            self._prim_instance = primitive(
                hyperparams=primitive_hyperparams(primitive_hyperparams.defaults(), **custom_hyperparams))
        else:  # is an instance
            self._prim_instance = primitive

        for labelIteration in range(self._iters):
            labeledIx = np.where(y.iloc[:, 0].values != '')[0]
            unlabeledIx = np.where(y.iloc[:, 0].values == '')[0]

            if (labelIteration == 0):
                num_instances_to_label = int(self._frac * len(unlabeledIx) + 0.5)

            labeledX = X.iloc[labeledIx]
            labeledy = y.iloc[labeledIx]

            self._prim_instance.set_training_data(inputs=labeledX, outputs=labeledy)
            self._prim_instance.fit()
            probas = self._prim_instance._clf.predict_proba(X.iloc[unlabeledIx])

            entropies = np.sum(np.log2(probas.clip(0.0000001, 1.0)) * probas, axis=1)
            # join the entropies and the unlabeled indecies into a single recarray and sort it by entropy
            entIdx = np.rec.fromarrays((entropies, unlabeledIx))
            entIdx.sort(axis=0)

            labelableIndices = entIdx['f1'][-num_instances_to_label:].reshape((-1,))

            predictions = self._prim_instance.produce(inputs=X.iloc[labelableIndices]).value
            ydf = y.iloc[labelableIndices, 0]
            ydf.loc[:] = predictions.iloc[:, 0]

        labeledIx = np.where(y.iloc[:, 0].values != '')[0]
        labeledX = X.iloc[labeledIx]
        labeledy = y.iloc[labeledIx]

        self._prim_instance.set_training_data(inputs=labeledX, outputs=labeledy)
        self._prim_instance.fit()
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
        output = self._prim_instance.produce(inputs=inputs)
        output = container.DataFrame(output.value, generate_metadata=True)
        return base.CallResult(output)
