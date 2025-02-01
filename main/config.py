import os
import os.path as osp
import sys
import numpy as np
import datetime
import yaml
import shutil
import glob
from easydict import EasyDict as edict


class Config:
    dataset = ''
    trainset_2d = ['MSCOCO' if dataset == 'MSCOCO' else 'MPII']
    testset = 'MSCOCO' if dataset == 'MSCOCO' else 'MPII'

    stage2 = {'NUM_CHANNELS': [32, 64], 'NUM_BLOCKS': [2, 2], 'NUM_BRANCHES': 2, 'NUM_MODULES': 1, 'FUSE_METHOD': 'SUM'}
    stage3 = {'NUM_CHANNELS': [32, 64, 128], 'NUM_BLOCKS': [2, 2, 2], 'NUM_BRANCHES': 3, 'NUM_MODULES': 4,
              'FUSE_METHOD': 'SUM'}
    stage4 = {'NUM_CHANNELS': [32, 64, 128, 256], 'NUM_BLOCKS': [2, 2, 2, 2], 'NUM_BRANCHES': 4, 'NUM_MODULES': 4,
              'FUSE_METHOD': 'SUM'}
    deconv = {'NUM_DECONVS': 1, 'NUM_CHANNELS': 32, 'KERNEL_SIZE': 4, 'NUM_BASIC_BLOCKS': 4}

    input_img_shape = (256, 256)
    output_hm_shape = (128, 128)
    sigma = 2
    target_type = 'gaussian'
    post_process = False
    use_target_weight = True
    use_different_joints_weight = False
    scale_factor = 0.35 if dataset == 'MSCOCO' else 0.25
    rot_factor = 45 if dataset == 'MSCOCO' else 30
    flip = True
    width_height_ratio = input_img_shape[1] / input_img_shape[0]
    lr_dec_epoch = [15]
    end_epoch = 20
    lr = 1e-5
    lr_dec_factor = 10
    train_batch_size = 16
    width_mult = 1.0
    depth_mult = 1.0
    hour_depth = 4
    hour_blocks = 2
    final_conv_kernel = 1
    init_weights = True
    select_data = False
    test_batch_size = 16
    use_gt_bbox = True
    nms_thre = 1.0
    soft_nms = False
    oks_thre = 0.9
    in_vis_thre = 0.2
    image_thre = 0.0
    num_thread = 1
    gpu_ids = '0'
    num_gpus = 1
    continue_train = False
    train_set = 'train2017'
    val_set = 'val2017'
    test_set = 'test-dev2017'
    prob_half_body = 0.3
    num_joints_half_body = 8
    data_format = 'jpg'
    rank = 0

    ## directory
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = osp.join(cur_dir, '..')
    data_dir = osp.join(root_dir, 'data')
    output_dir = osp.join(root_dir, 'output')

    save_folder = 'exp_' + str(datetime.datetime.now())[5:-10]
    save_folder = save_folder.replace(" ", "_")
    output_dir = osp.join(output_dir, save_folder)
    print('output dir: ', output_dir)

    model_dir = osp.join(output_dir, 'checkpoint')
    vis_dir = osp.join(output_dir, 'vis')
    log_dir = osp.join(output_dir, 'log')
    result_dir = osp.join(output_dir, 'result')

    def set_args(self, gpu_ids, continue_train=False, is_test=False, exp_dir=''):
        self.gpu_ids = gpu_ids
        self.num_gpus = len(self.gpu_ids.split(','))

        if not is_test:
            self.continue_train = continue_train
            if self.continue_train:
                if exp_dir:
                    checkpoints = sorted(glob.glob(osp.join(exp_dir, 'checkpoint') + '/*.pth.tar'),
                                         key=lambda x: int(x.split('_')[-1][:-8]))
                    shutil.copy(checkpoints[-1], osp.join(cfg.model_dir, checkpoints[-1].split('/')[-1]))

                else:
                    shutil.copy(osp.join(cfg.root_dir, 'tool', 'snapshot_0.pth.tar'),
                                osp.join(cfg.model_dir, 'snapshot_0.pth.tar'))
        elif is_test and exp_dir:
            self.output_dir = exp_dir
            self.model_dir = osp.join(self.output_dir, 'checkpoint')
            self.vis_dir = osp.join(self.output_dir, 'vis')
            self.log_dir = osp.join(self.output_dir, 'log')
            self.result_dir = osp.join(self.output_dir, 'result')

        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        print('>>> Using GPU: {}'.format(self.gpu_ids))

    def update(self, config_file):
        with open(config_file) as f:
            exp_config = edict(yaml.load(f, Loader=yaml.FullLoader))
            for k, v in exp_config.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
                else:
                    raise ValueError("{} not exist in config.py".format(k))


cfg = Config()

sys.path.insert(0, osp.join(cfg.root_dir, 'common'))
from utils.dir import add_pypath, make_folder

add_pypath(osp.join(cfg.data_dir))
dataset_list = ['MSCOCO']
for i in range(len(dataset_list)):
    add_pypath(osp.join(cfg.data_dir, dataset_list[i]))
make_folder(cfg.model_dir)
make_folder(cfg.vis_dir)
make_folder(cfg.log_dir)
make_folder(cfg.result_dir)
