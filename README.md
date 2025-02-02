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

### Data preparation

Please refer to the data folder for dataset downloads.

### Training and Testing
Make sure you are in the main folder.

```
cd ${POSE_ROOT/main}
```

#### Traing on COCO train2017 dataset
```
python train.py --gpu 0 --cfg ../assets/coco.yml 
```

#### Testing on COCO val2017 dataset
```
python test.py --gpu 0 --cfg ../assets/coco.yml  --exp_dir ../output/exp_{}-{}_{}:{} --test_epoch {}
```

If you want to obtain the test results for each epoch, please run:
```
./run_coco_epochs.sh
```

#### Traing on MPII train dataset
```
python train.py --gpu 0 --cfg ../assets/mpii.yml 
```

#### Testing on MPII test dataset
```
python test.py --gpu 0 --cfg ../assets/mpii.yml  --exp_dir ../output/exp_{}-{}_{}:{} --test_epoch {}
```

After the code execution is completed, you will obtain the prediction file `pred.mat`. Finally, please follow the evaluation procedure provided on the website below to obtain the final evaluation metrics:

[Website:](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/software-and-datasets/mpii-human-pose-dataset/evalution)

This process will allow you to calculate the final evaluation results based on the `pred.mat` file.




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





