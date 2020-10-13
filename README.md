# VVSec Overview

VVSec is a novel volumetric video security mechanism, which makes benign use of adversarial perturbations to obfuscate 
the security and privacy sensitive 3D face models. Such obfuscation ensures that the 3D models cannot be exploited to 
bypass deep learning-based face authentications. Meanwhile, the injected perturbations are not perceivable by the 
end-users, maintaining the original quality of experience in volumetric video streaming.

To find more details, please read our paper:

Zhongze Tang, Xianglong Feng, Yi Xie, Huy Phan, Tian Guo, Bo Yuan, and Sheng Wei. 2020. VVSec: Securing Volumetric Video Streaming via Benign Use of Adversarial Perturbation. In Proceedings of the 28th ACM International Conference on Multimedia (MM '20). Association for Computing Machinery, New York, NY, USA, 3614–3623. DOI: https://doi.org/10.1145/3394171.3413639

This repository contains the source code and the dataset to test our VVSec, which adds perturbation to a human face in a 
volumetric video so that the face and its corresponding depth can no longer be used to bypass a deep learning-based face 
authentication system.

# Quick Start

## Download Datasets

Dataset \#1 has already been included in this repo, under ```./dataset/ds1/```.

Dataset \#2 can be found at https://vap.aau.dk/rgb-d-face-database/.

You can check [Datasets](#Datasets) section for details.

## Download Model

Download the model of the face authentication system 
from https://drive.google.com/file/d/16v77oJZafuFbw8_vuoQ2jwmx0_N_Q6y5/view?usp=sharing,
and put it under ```./model/```.

You can check [Face Authentication System](#face-authentication-system) section for details.

## Setup the Environment

We recommend using [Anaconda](https://www.anaconda.com/products/individual) to setup the environment.

Install the following libraries:

- Python 3.7
- keras 2.2.4
- TensorFlow 1.14.0
- Pillow 6.2.1
- numpy 1.17.4
- matplotlib 3.3.1

## Give it a try!

Simply run ```main.py``` to see the results!

If you want to change the inputs, just simply modify the path of the reference and user inputs files.

These two figures match Case 1 and 9 of Table 1 in our paper, respectively.

![](/pic/demo_result_1.png)

![](/pic/demo_result_2.png)

# Repository Hierarchy

```
.
├── adv_attack.py // Adversarial attack algorithm
├── dataset
│   ├── ds1 // Dataset 1, which comes from a Volumetric Video
│   │   ├── 0.jpg_c.bmp // RGB image used for VVSec
│   │   ├── 0.jpg_d.bmp // Visualized depth data
│   │   ├── 0.jpg_d.dat // Depth data used for VVSec
│   │   ├── ...
│   └── ds2 // Dataset 2, the RGB-D dataset
│       ├── faceid_train
│       │   ├── (2012-05-16)(151751)
│       │   │   ├── 001_1_c.bmp // RGB image used for VVSec
│       │   │   ├── 001_1_d.dat // Depth data used for VVSec
│       │   │   ├── 001_2_c.bmp
│       │   │   ├── 001_2_d.dat
│       │   │   ...
│       │   ...
│       ├── faceid_val
│       │   ├── (2012-05-18)(152717)
│       │   │   ├── 001_1_c.bmp
│       │   │   ├── 001_1_d.dat
│       │   │   ├── 001_2_c.bmp
│       │   │   ├── 001_2_d.dat
│       │   │   ...
│       │   ...
│       └── PUT_DATASET2_HERE
├── main.py // Main program, quickly try VVSec
├── model
│   └── model_top.model // Model for the face authentication used in VVSec, you have to download it
├── pic
│   ├── demo_result_1.png
│   └── demo_result_2.png
├── README.md // This file
├── utils.py // Some helper functions
└── .gitignore

```

# Datasets

For both datasets, the RGB images are stored as a 32-bit bitmap (```*_c.bmp```) with resolution 1280 × 960px.

The depth images are stored as plaintext files (```*_d.dat```), where a depth value represents the corresponding depth 
pixel, with resolution 640×480px. The numbers in the depth data files are distance in millimetres from the sensor. 
Other values are constants specified by the Kinect:

-   -1 - Undefined distance 
-    0 - Too near  
- 4095 - Too far

## Dataset \#1

Dataset #1 contains 11 RGB-D images of one user extracted from
the volumetric video demo in the [Depthkit](https://www.depthkit.tv/).

To download the volumetric video, you need to log into DepthKit to visit https://www.depthkit.tv/downloads, then 
download the ```Sample Project```.

We also provide visualized depth information of all the data in Dataset \#1, which ends in ```*_d.bmp```.

## Dataset \#2

It comes from the paper 
[An RGB-D Database Using Microsoft's Kinect for Windows for Face Detection](https://ieeexplore.ieee.org/document/6395071),
 and its original download webpage is https://vap.aau.dk/rgb-d-face-database/.

# Face Authentication System

We adopt an open-source [3D image-based face authentication system]((https://towardsdatascience.com/how-i-implemented-iphone-xs-faceid-using-deep-learning-in-python-d5dbaa128e1d)) 
to validate the attack and defense mechanisms presented in this [work](https://towardsdatascience.com/how-i-implemented-iphone-xs-faceid-using-deep-learning-in-python-d5dbaa128e1d).

We provide the file of the DNN model in this Siamese Network, which can be downloaded from
https://drive.google.com/file/d/16v77oJZafuFbw8_vuoQ2jwmx0_N_Q6y5/view?usp=sharing, and should be put under ```./model/```.
To get the complete model of 
this face authentication system, call ```get_model()``` function in ```utils.py```.


# Cite Our Work

```bibtex
@inproceedings{tang20vvsec,
  title = {VVSec: Securing Volumetric Video Streaming via Benign Use of Adversarial Perturbation},
  author = {Tang, Zhongze and Feng, Xianglong and Xie, Yi and Phan, Huy and Guo, Tian and Yuan, Bo and Wei, Sheng},
  booktitle = {ACM International Conference on Multimedia (MM)},
  pages = {3614--3623},
  year = {2020}
}
```

# Contact

If you have any questions or ideas to discuss, please contact Zhongze Tang  (zhongze.tang@rutgers.edu). Thank you!

# License

MIT