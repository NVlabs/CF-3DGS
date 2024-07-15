<!-- PROJECT LOGO -->

<p align="center">

  <h1 align="center">COLMAP-Free 3D Gaussian Splatting</h1>
  <p align="center">
    <a href="https://oasisyang.github.io/"><strong>Yang Fu</strong></a>
    ·
    <a href="https://sifeiliu.net/"><strong>Sifei Liu</strong></a>
    ·
    <a><strong>Amey Kulkarni</strong></a>
    ·
    <a href="https://jankautz.com/"><strong>Jan Kautz</strong></a>
    <br>
    <a href="https://people.eecs.berkeley.edu/~efros/"><strong>Alexei A. Efros</strong></a>
    ·
    <a href="https://xiaolonw.github.io/"><strong>Xiaolong Wang</strong></a>
  </p>
  <h3 align="center"><a href="https://arxiv.org/abs/2312.07504">Paper</a> | <a href="https://youtu.be/mGeVQS4ExK4?si=kqu3dbXopDdNg0wX">Video</a> | <a href="https://oasisyang.github.io/colmap-free-3dgs/">Project Page</a></h3>
  <div align="center"></div>
</p>

<p align="center">
  <a href="">
    <img src="./assets/teaser.gif" alt="Logo" width="100%">
  </a>
</p>

## Installation

##### (Recommended)
The codes have been tested on python 3.10, CUDA>=11.6. The simplest way to install all dependences is to use [anaconda](https://www.anaconda.com/) and [pip](https://pypi.org/project/pip/) in the following steps: 

```bash
conda create -n cf3dgs python=3.10
conda activate cf3dgs
conda install conda-forge::cudatoolkit-dev=11.7.0
conda install pytorch==2.0.0 torchvision==0.15.0 pytorch-cuda=11.7 -c pytorch -c nvidia
git clone --recursive git@github.com:NVlabs/CF-3DGS.git
pip install -r requirements.txt
```

## Dataset Preparsion
DATAROOT is `./data` by default. Please first make data folder by `mkdir data`.

### Tanks and Temples

Download the data preprocessed by [Nope-NeRF](https://github.com/ActiveVisionLab/nope-nerf/?tab=readme-ov-file#Data) as below, and the data is saved into the `./data/Tanks` folder.
```bash
wget https://www.robots.ox.ac.uk/~wenjing/Tanks.zip
```

### CO3D
Download our preprocessed [data](https://ucsdcloud-my.sharepoint.com/:u:/g/personal/yafu_ucsd_edu/EftJV9Xpn0hNjmOiGKZuzyIBW5j6hAVEGhewc8aUcFShEA?e=x1aXVx), and put it saved into the `./data/co3d` folder.


## Run

### Training
```bash
python run_cf3dgs.py -s data/Tanks/Francis \ # change the scene path
                     --mode train \
                     --data_type tanks
```

### Evaluation
```bash
# pose estimation
python run_cf3dgs.py --source data/Tanks/Francis \
                     --mode eval_pose \
                     --data_type tanks \
                     --model_path ${CKPT_PATH} 
# by default the checkpoint should be store in "./output/progressive/Tanks_Francis/chkpnt/ep00_init.pth"
# novel view synthesis
python run_cf3dgs.py --source data/Tanks/Francis \
                     --mode eval_nvs \
                     --data_type tanks \
                     --model_path ${CKPT_PATH} 
``` 
We release some of the novel view synthesis results ([gdrive](https://drive.google.com/drive/folders/1p3WljCN90zrm1N5lO-24OLHmUFmFWntt?usp=sharing)) for comparison with future works.

### Run on your own video

* To run CF-3DGS on your own video, you need to first convert your video to frames and save them to `./data/$CUSTOM_DATA/images/
`

* Camera intrincics can be obtained by running COLMAP (check details in `convert.py`). Otherwise, we provide a heuristic camera setting which should work for most landscope videos. 

* Run the following commands:

```bash
python run_cf3dgs.py -s ./data/$CUSTOM_DATA/ \ # change to your data path
                     --mode train \
                     --data_type custom
```

## Acknowledgement
Our render is built upon [3DGS](https://github.com/graphdeco-inria/gaussian-splatting). The data processing and visualization codes are partially borrowed from [Nope-NeRF](https://github.com/ActiveVisionLab/nope-nerf/). We thank all the authors for their great repos.

## Citation

If you find this code helpful, please cite:

```
@InProceedings{Fu_2024_CVPR,
    author    = {Fu, Yang and Liu, Sifei and Kulkarni, Amey and Kautz, Jan and Efros, Alexei A. and Wang, Xiaolong},
    title     = {COLMAP-Free 3D Gaussian Splatting},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {20796-20805}
}
```