import os
import warnings

import autonbox
import cv2
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


class Hyperparams(hyperparams.Hyperparams):
    num_workers = hyperparams.Hyperparameter[int](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ResourcesUseParameter'],
        default=0,
        description='The number of subprocesses to use for data loading. 0 means that the data will be loaded in the '
                    'main process.'
    )


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
    no_cuda = True
    mean = [114.7748, 107.7354, 99.4750]
    batch_size = 32


class ResNext101KineticsPrimitive(FeaturizationTransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    Video Feature Extraction for Action Classification With 3D ResNet
    """
    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '15935e70-0605-4ded-87cf-2933ca35d4dc',
            'version': '0.2.0',
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
                'file_uri': 'http://public.datadrivendiscovery.org/resnext-101-kinetics.pth',
                'file_digest': autonbox.__digest_static_file_resnext__
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

        torch.manual_seed(self.random_seed)  # seed the RNG for all devices (both CPU and CUDA):

        # Use GPU if available
        if torch.cuda.is_available():
            self.logger.info("Use GPU.")
            self._config.no_cuda = False
            # For reproducibility on CuDNN backend
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            self.logger.info("Use CPU.")
            self._config.no_cuda = True

        self.logger.info('Number of workers: {}'.format(self.hyperparams['num_workers']))

        self._down_rate = 1

    def _instantiate_model(self):
        model = generate_model(self._config)
        model_data = self._load_model()
        if self._config.no_cuda:
            state_dict = {k.replace('module.', ''): v for k, v in model_data['state_dict'].items()}  # for cpu only
        else:
            state_dict = model_data['state_dict']
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        :param inputs: assume the first column is the filename
        :param timeout:
        :param iterations:
        :return:
        """
        model = self._instantiate_model()
        features = []
        # TODO consider a more robust means to 1) get location_base_uris and remove file://
        media_root_dir = inputs.metadata.query((0, 0))['location_base_uris'][0][len('file://'):]  # remove file://
        for filename in inputs.iloc[:, 0]:
            file_path = os.path.join(media_root_dir, filename)
            if os.path.isfile(file_path):
                video = self._read_fileuri(file_path)  # video is a ndarray of F x H x W x C, e.g. (408, 240, 320, 3)
                feature = self._generate_vid_feature(model, video)
            else:
                self.logger.warning("No such file {}. Feature vector will be set to all zeros.".format(file_path))
                feature = np.zeros(2048)
            features.append(feature)

        results = container.DataFrame(features, generate_metadata=True)

        return base.CallResult(results)

    def _generate_vid_feature(self, model, vid_matrix):
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
                                                  num_workers=self.hyperparams['num_workers'],
                                                  shuffle=False, pin_memory=True)
        video_outputs = []
        with torch.no_grad():
            for i, inputs in enumerate(data_loader):
                inputs = Variable(inputs)
                # input is of shape n x 3 x sample_duration x 112 x 112
                outputs = model(inputs)
                # output is of format n(batch size) x d(dimension of feature)
                video_outputs.append(outputs.cpu().data)
        video_outputs = np.concatenate(video_outputs, axis=0)
        mean_feature = np.mean(video_outputs, axis=0)  # shape of (d, )
        return mean_feature

    def _load_model(self):
        """
        Loads the model from the volume
        :return:
        """
        key_filename = autonbox.__key_static_file_resnext__
        static_dir = os.getenv('D3MSTATICDIR', '/static')
        if key_filename in self.volumes:
            _weight_file_path = self.volumes[key_filename]
            self.logger.info("Weights file path found in static volumes")
        else:
            self.logger.info("Trying to locate weights file in the static folder {}".format(static_dir))
            _weight_file_path = os.path.join(static_dir, autonbox.__digest_static_file_resnext__)

        if os.path.isfile(_weight_file_path):
            if torch.cuda.is_available():  # GPU
                model_data = torch.load(_weight_file_path)
            else:  # CPU only
                model_data = torch.load(_weight_file_path, map_location='cpu')
            self.logger.info("Loaded weights file")
        else:
            raise ValueError("Can't get weights file from the volume by key: {} or in the static folder: {}".format(
                key_filename, static_dir))

        return model_data

    def _read_fileuri(self, fileuri: str) -> container.ndarray:
        """
        @see https://gitlab.com/datadrivendiscovery/common-primitives/blob/master/common_primitives/video_reader.py#L65
        :param fileuri:
        :return:
        """
        capture = cv2.VideoCapture(fileuri)
        frames = []

        try:
            while capture.isOpened():
                ret, frame = capture.read()
                if not ret:
                    break
                else:
                    assert frame.dtype == np.uint8, frame.dtype

                    if frame.ndim == 2:
                        # Make sure there are always three dimensions.
                        frame = frame.reshape(list(frame.shape) + [1])

                    assert frame.ndim == 3, frame.ndim

                    frames.append(frame)
        finally:
            capture.release()

        return container.ndarray(np.array(frames), generate_metadata=False)
