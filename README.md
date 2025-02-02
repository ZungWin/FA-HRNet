# Enhanced 2D Human Pose Estimation via Feature-Aligned High-Resolution Network

This repository is the offical Pytorch implementation of _Enhanced 2D Human Pose Estimation via Feature-Aligned High-Resolution Network_, We will release our code in the near future.
## News
[2025.01.15] Create repository

[2025.01.29] Add files: (1)common/nets/module.py ;(2)common/nets/loss.py ;(3)common/nets/layer.py

[2025.01.30] Add files: (1)common/utils/preprocessing.py ;(2)common/utils/inference.py ;(3)common/logger.py ;(4)common/timer.py and data/

[2025.02.01] Add files: (1)main/ ;(2)assets/

## Introduction


## Quick start

### Enivornment installation
1. Install pytorch >= v2.1.1 following [official instruction](https://pytorch.org/).

2. Clone this repo, and we'll call the directory that you cloned as ${POSE_ROOT}.

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Install [COCOAPI](https://github.com/cocodataset/cocoapi):
   ```
   # COCOAPI=/path/to/clone/cocoapi
   git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
   cd $COCOAPI/PythonAPI
   # Install into global site-packages
   make install
   # Alternatively, if you do not have permissions or prefer
   # not to install the COCO API into global site-packages
   python3 setup.py install --user
   ```
5. Init output(training model output directory):
   
   ```
   mkdir output
   ```
   Your directory tree should look like this:

   ```
   ${POSE_ROOT}
   ├── assets
   ├── common
   ├── data
   ├── main
   ├── output
   ├── README.md
   └── requirements.txt
   ```
6. Download pretrained models from our model zoo[provided after the publication of the paper]

## Citation
If you use our code or models in your research, please cite with:
```
@inproceedings{zhu2025FAHRNet,
  title={Enhanced 2D Human Pose Estimation via Feature-Aligned High-Resolution Network},
  author={Yuhe, Zhu and Zhangwen, Lyu and Rong, Liu and Yinwei, Zhan},
  booktitle={The Visual Computer},
  year={2025}
}
```





