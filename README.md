<div align="center">

<h1>Keypoint Fusion for RGB-D Based 3D Hand Pose Estimation [AAAI2024]</h1>



<h4 align="center">
  <a href="https://doi.org/10.1609/aaai.v38i4.28166" target='_blank'>[Paper Page]</a> 
</h4>

</div>

<div>

## Setup with Conda
```bash
# create conda env
conda create -n dir python=3.9
# install torch
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
# install other requirements
git clone --recursive https://github.com/ru1ven/KeypointFusion.git
cd KeypointFusion
pip install -r ./requirements.txt
```

## Dataset preparation
Download the [DexYCB dataset](https://dex-ycb.github.io/) and the [annotations](https://drive.google.com/drive/folders/1YAF1jAsGi2aWkTml1tFV2y39aSmIYpde?usp=sharing).

## Training & Evaluation
Download is our [pre-trained model](https://drive.google.com/file/d/1sl0r62C8c1eYlFKyFGk-CTW2hoXFvqIa/view?usp=sharing) on DexYCB. 

```bash
python train.py
```
you would get the following output:
```bash

[mean_Error 6.927]
[PA_mean_Error 4.790]
```

Comparison on HO3D can be seen in [here](https://codalab.lisn.upsaclay.fr/competitions/4318#results).



## BibTeX


```
@inproceedings{liu2024keypoint,
  title={Keypoint Fusion for RGB-D Based 3D Hand Pose Estimation},
  author={Liu, Xingyu and Ren, Pengfei and Gao, Yuanyuan and Wang, Jingyu and Sun, Haifeng and Qi, Qi and Zhuang, Zirui and Liao, Jianxin},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={4},
  pages={3756--3764},
  year={2024}
}

```
<div>
