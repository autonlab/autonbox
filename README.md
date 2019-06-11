# The Auton Lab TA1 primitives

This repository contains additional Auton Lab TA1 primitives for the D3M program.

1. [`Iterative Labeling`](autonbox/iterative_labeling.py) - Blackbox based iterative labeling for semi-supervised learning
1. [`Video featurizer`](autonbox/resnext101_kinetics_video_features.py) - Video Feature Extraction for Action Classification With 3D ResNet

## Installation
To install primitives, run:
```bash
pip install -U -e git+https://github.com/autonlab/autonbox.git#egg=autonbox
```

`Video featurizer` requires a static file, pre-trained model weights.
To download it, run: 
```bash
mkdir -p /tmp/cmu/pretrained_files
python3 -m d3m index download -o /tmp/cmu/pretrained_files # requires d3m core
```

## Video featurizer
The primitive outputs a data frame of size N x M, where N is the number of videos and M is 2024 features of type float.

It supports running on GPUs.
