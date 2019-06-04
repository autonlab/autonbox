import logging
import os

import autonbox
import numpy as np
import torch
import typing
from autonbox.contrib.resnet.dataset import Video
from autonbox.contrib.resnet.model import generate_model
from autonbox.contrib.resnet.spatial_transforms import (Compose, Normalize, Scale, CenterCrop, ToTensor)
from autonbox.contrib.resnet.temporal_transforms import LoopPadding
from d3m import container, utils as d3m_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base
from d3m.primitive_interfaces.base import CallResult
from d3m.primitive_interfaces.featurization import FeaturizationTransformerPrimitiveBase
from torch.autograd import Variable

Inputs = container.DataFrame
Outputs = container.DataFrame

_logger = logging.getLogger(__name__)


class Hyperparams(hyperparams.Hyperparams):
    pass


class ResNext101KineticsParams(object):
    # TODO determine which to make as hyper-parameters
    n_classes = 400
    mode = 'feature'
    clip_vid = 'mean'
    down_rate = 1
    model_name = 'resnext'
    model_depth = 101
    resnet_shortcut = 'B'
    resnext_cardinality = 32
    sample_size = 112
    sample_duration = 16  # number of frames in one clip
    no_cuda = False
    mean = [114.7748, 107.7354, 99.4750]
    batch_size = 32
    n_threads = 4


class ResNext101KineticsPrimitive(FeaturizationTransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    Video Feature Extraction for Action Classification With 3D ResNet
    """
    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '15935e70-0605-4ded-87cf-2933ca35d4dc',
            'version': '0.1.0',
            "name": "Video Feature Extraction for Action Classification With 3D ResNet",
            'description': "Video Feature Extraction for Action Classification With 3D ResNet",
            'python_path': 'd3m.primitives.feature_extraction.resnext101_kinetics_video_features.VideoFeaturizer',
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
            }, {
                'type': metadata_base.PrimitiveInstallationType.FILE,
                'key': autonbox.__key_static_file_resnext__,
                'file_uri': 'https://doc-0k-74-docs.googleusercontent.com/docs/securesc/ha0ro937gcuc7l7deffksulhg5h7mbp1/qg2eg2c7l2nt9iglpq74jba810p0br5n/1559656800000/09020394064798262542/*/1cULocPe5YvPGWU4tV5t6fC9rJdGLfkWe?e=download',
                'file_digest': 'f82e4e519723fc7b2ff3761ea35600bdaf796fb7a4e62ee4c5591da7ffe48326'
            }],
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.CONVOLUTIONAL_NEURAL_NETWORK,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.FEATURE_EXTRACTION,
        },
    )

    def __init__(self, *, hyperparams: Hyperparams, volumes: typing.Union[typing.Dict[str, str], None] = None) -> None:
        super().__init__(hyperparams=hyperparams, volumes=volumes)

        self._config = ResNext101KineticsParams
        self._model = generate_model(self._config)
        self._down_rate = 1
        model_data = self._load_model()
        if self._config.no_cuda:
            state_dict = {k.replace('module.', ''): v for k, v in model_data['state_dict'].items()}  # for cpu only
        else:
            state_dict = model_data['state_dict']
        self._model.load_state_dict(state_dict)
        self._model.eval()
        _logger.info(self._model)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        # inputs is DataFrame
        # inputs.iloc[0,0] is a ndarray of size (408, 240, 320, 3)

        features = []
        for video in inputs.iloc[:, 0]:
            features.append(self._generate_vid_feature(video))

        results = container.DataFrame(features, generate_metadata=True)

        return base.CallResult(results)

    def _generate_vid_feature(self, vid_matrix):
        """
        Modified from function classify_video()
        :param vid_matrix: takes in video matrix F(frames) x H(height) x W(width) x C(channels)
                           dtype of matrix is uint8
        :return: ndarray representation of video
        """
        assert vid_matrix.ndim == 4 and self._down_rate <= 1  # sanity check
        spatial_transform = Compose([Scale(self._config.sample_size),
                                     CenterCrop(self._config.sample_size),
                                     ToTensor(),
                                     Normalize(self._config.mean, [1, 1, 1])])
        temporal_transform = LoopPadding(self._config.sample_duration)
        data = Video(vid_matrix, spatial_transform=spatial_transform,
                     temporal_transform=temporal_transform,
                     sample_duration=self._config.sample_duration, down_rate=self._down_rate)
        data_loader = torch.utils.data.DataLoader(data, batch_size=self._config.batch_size,
                                                  shuffle=False, num_workers=self._config.n_threads, pin_memory=True)
        video_outputs = []
        video_segments = []
        with torch.no_grad():
            for i, (inputs, segments) in enumerate(data_loader):
                inputs = Variable(inputs)
                # input is of shape n x 3 x sample_duration x 112 x 112
                outputs = self._model(inputs)
                # output is of format n(batch size) x d(dimension of feature)
                video_outputs.append(outputs.cpu().data)
                video_segments.append(segments)
                # segments is of shape batch_size x 2
        video_outputs = np.concatenate(video_outputs, axis=0)
        # video_segments = np.concatenate(video_segments,axis=0)
        mean_feature = np.mean(video_outputs, axis=0)  # shape of (d, )
        return mean_feature

    def _load_model(self):
        """
        Loads the model from the volume
        :return:
        """
        key_filename = autonbox.__key_static_file_resnext__
        if key_filename in self.volumes:
            self._weight_file_path = self.volumes[key_filename]
            _logger.info("Weights file found in static volumes")
            model_data = torch.load(self._weight_file_path)
        else:
            raise ValueError("Can't get weights file from the volume by key: {}".format(key_filename))

        return model_data
