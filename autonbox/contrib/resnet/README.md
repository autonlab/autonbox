## Video Feature Extraction for Action Classification With 3D ResNet

* This repo is forked from [this work](https://github.com/kenshohara/video-classification-3d-cnn-pytorch) and added 
with changes to run feature extraction from videos
* This method is based on 3D ResNet trained by [this work](https://github.com/kenshohara/3D-ResNets-PyTorch)

## Citation
If you use this code, please cite the original paper:
```
@article{hara3dcnns,
  author={Kensho Hara and Hirokatsu Kataoka and Yutaka Satoh},
  title={Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet?},
  journal={arXiv preprint},
  volume={arXiv:1711.09577},
  year={2017},
}
```

## Requirements
* [PyTorch](http://pytorch.org/) version0.3
* FFmpeg, FFprobe if need video processing
* Python 3
* Pillow for frame image processing

## Before feature extraction
* Download pre-trained models into ```$MODEL_DIR``` folder
* Prepare video features as numpy arrays with shape ```F x H x W x C``` per video in ```$VIDEO_ROOT```, where 
F is frame number, H and W are height and width of videos and C is number of channels (3 for RGB)
* Prepare the list of videos(paths) in ```$LIST_DIR```
* If videos are stored in form of jpg files, run ```python generate_matrix.py $jpg_root $dst_npy_root``` to 
generate numpy matrices.

## Featrue extraction
* Run following for features extraction (this script calls ```extract_feature.py``` with options). Specify output 
directory in ```$OUT_DIR``` and a json file name 
```
bash run_extract_module.sh
``` 
* Function in ```extract_feature.py``` will take in video matrices and output a json file containing feature vectors 
of dimension 2048, for details see function ```generate_vid_feature```. 
* Make sure option ```n_classes``` in the script aligns with pre-trained model of choice. For instance, kinetics 
dataset has ```n_classes=400```, HMDB dataset has ```n_classes=51```.
* The feature for each video has dimension 2048.

## Embedding Visualization
* To visualize features using TSNE embedding, run
```
python visualize_features.py \path_to_json \path_to_video_labels
```
output:
![TSNE](https://github.com/MYusha/video-classification-3d-cnn-pytorch/blob/master/Figure_1.png)



