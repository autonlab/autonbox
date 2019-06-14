import copy
import json
import os

import functools
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def video_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, 'image_{:05d}.jpg'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video

    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            if subset == 'testing':
                video_names.append('test/{}'.format(key))
            else:
                label = value['annotations']['label']
                video_names.append('{}/{}'.format(label, key))
                annotations.append(value['annotations'])

    return video_names, annotations


def make_dataset(video_path, sample_duration, down_rate):
    dataset = []

    n_frames = len(os.listdir(video_path))

    begin_t = 1
    end_t = n_frames
    sample = {
        'video': video_path,
        'segment': [begin_t, end_t],
        'n_frames': n_frames,
    }

    step = sample_duration * down_rate
    for i in range(1, (n_frames - sample_duration + 1), step):
        sample_i = copy.deepcopy(sample)
        sample_i['frame_indices'] = list(range(i, i + sample_duration))
        sample_i['segment'] = torch.IntTensor([i, i + sample_duration - 1])
        dataset.append(sample_i)

    return dataset

def make_dataset_from_matrix(vid_matrix, sample_duration, down_rate):
    dataset = []
    n_frames = vid_matrix.shape[0]
    begin_t = 1
    end_t = n_frames
    sample = {
        'video': vid_matrix,
        'segment': [begin_t, end_t],
        'n_frames': n_frames,
    }
    step = sample_duration * down_rate
    # for i in range(1, n_frames - sample_duration + 1, step):
    #     sample_i = copy.deepcopy(sample)
    #     sample_i['frame_indices'] = list(range(i, i + sample_duration))
    #     sample_i['segment'] = torch.IntTensor([i, i + sample_duration - 1])
    #     dataset.append(sample_i)
    for j in range(1, n_frames, step):
        sample_j = copy.deepcopy(sample)
        sample_j['frame_indices'] = list(range(j, min(n_frames + 1, j + sample_duration)))
        dataset.append(sample_j)

    return dataset



class Video(data.Dataset):
    def __init__(self, vid_content,
                 spatial_transform=None, temporal_transform=None,
                 sample_duration=16, get_loader=get_default_video_loader, down_rate=1):
        if isinstance(vid_content, str):
            self.data = make_dataset(vid_content, sample_duration, down_rate)
        elif isinstance(vid_content,np.ndarray):
            self.data = make_dataset_from_matrix(vid_content, sample_duration, down_rate)
        else:
            print('do not support input to class Video with type {}'.format(type(vid_content)))

        # data is list of dicts with keys etc."frame_indicies","segments"
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.loader = get_loader()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        vid_content = self.data[index]['video']

        frame_indices = self.data[index]['frame_indices']
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        if isinstance(vid_content,str):
            clip = self.loader(vid_content, frame_indices)
        elif isinstance(vid_content,np.ndarray):
            # clip = vid_content[frame_indices, :, :, :]
            clip = [Image.fromarray(vid_content[j-1,:,:,:]) for j in frame_indices]
        # a list(len=16) of PIL images!
        # can also take numpy.ndarray (H x W x C) instead
        if self.spatial_transform is not None:
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
        # is originally 3x16x112x112
        # target = self.data[index]['n_frames']

        return clip # , target

    def __len__(self):
        return len(self.data)
