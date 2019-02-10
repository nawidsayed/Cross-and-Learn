# Cross and Learn: Cross-Modal Self-Supervision

This repository contains parts of the code used to produce the action recognition results for my GCPR 2018 paper, which can be found [here](https://arxiv.org/pdf/1811.03879.pdf). 

The file `net_features.pkl` contains the parameters (conv layers) of my pre-trained CaffeNet for comparison purposes. It can be used via the file `model.py` which is independent from the other parts of the framework. The input tensors of this model should be globally normalized to have zero mean and unit variance.


### Requirements

The requirements can be found in requirements.txt and can be installed via `pip install -r requirements.txt`

### Data preparation UCF-101

The framework for data loading can be found in the class `UCF101_i` in `compvis/datasets/ds_info.py` this class can be modified to fit your own needs as long as it implements all the methods required in the parent class `Base_Info_Video`. 

If unmodified, the following directory structure for the dataset is expected, `path_ucf` can be set in `config.yml`.

```
path_ucf
│   dict_names.pkl 
│   dict_norms.pkl
│   dict_mags.pkl (optional)
│
└───rgb
│   └───ucf101_0
│   └───ucf101_1
│       ...
│   └───ucf101_13319
│
└───flow
│   └───x
│   │   └───ucf101_0
│   │   └───ucf101_1
│   │       ...
│   │    
│   └───y
│       └───ucf101_0
│       └───ucf101_1
│           ...
│
└───ucfTrainTestlist
    │   trainlist01.txt
    │   trainlist02.txt
        ...

```
The subfolders `ucf101_0, ucf101_1, ..., ucf101_13319` contain the rgb and flow frames of the respective videos in the dataset stored as jpg images, the optical flow images are grayscale images. The jpg files are named and numbered in the following manner:

```
ucf101_0
│   frame00001.jpg
│   frame00002.jpg
    ...
```
The file `dict_names.pkl` is a serialized dictionary which translates the video names to folder numberings, some exemplary `key: val` pairs look like this: `'v_ThrowDiscus_g18_c01.avi': 'ucf101_999'` and this `'v_Mixing_g16_c04.avi': 'ucf101_984'` . The correct identification of the video names is necessary in order to split the data according to the train/test splits in the folder `ucfTrainTestlist`. It is also necessary for obtaining the labels for fine-tuning. The folder `ucfTrainTestlist` can be downloaded and extracted from the main website of UCF-101 http://crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip .

The file `dict_norms.pkl` is a serialized dictionary containing the normalization factors for all optical flow frames. The keys of the dictionary are given by `ucf101_0, ucf101_1, ..., ucf101_13319` and the value for each key is a 1D numpy array with its length being equal to the number of frames in the video. Each element in such an array contains the normalization factor for the repsective frame in the video. This dictionary is necessary in order to retreive the original normalization of the optical flow during training as the optical flow frames are usually normalized before being stored onto the hard drive.

Lastly the optional file `dict_mags.pkl` is a serialized dictionary containing very similar to `dict_norms.pkl` but containing the average magnitude of each flow frame (note that the magnitude is not just the logarithm of the normalization factor). The keys of the dictionary are given by `ucf101_0, ucf101_1, ..., ucf101_13319` and the value for each key is a 1D numpy array with its length being equal to the number of frames in the video. If this file is not present it will be automatically generated and saved, which however can take several hours (but only once).

### Usage

To be extended soon



