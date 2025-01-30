import os
import os.path as osp
import numpy as np
from collections import OrderedDict

from config import cfg
import copy
import json
from scipy.io import loadmat, savemat
import cv2
import random
import torch
from utils.preprocessing import load_img, affine_transform, get_affine_transform_by_center

import logging

logger = logging.getLogger(__name__)

class MPII(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        self.pixel_std = 200
        self.transform = transform
        self.data_split = data_split
        self.img_path = osp.join('..', 'data', 'MPII', 'data', 'images')
        self.annot_path = osp.join('..', 'data', 'MPII', 'data', 'annotations')

        self.train_gt_file = 'train.json'
        self.train_gt_path = os.path.join(self.annot_path, self.train_gt_file)

        self.val_gt_file = 'valid.json'
        self.val_gt_path = os.path.join(self.annot_path, self.val_gt_file)
        self.val_gt_mat = os.path.join(self.annot_path, 'mpii_gt_val.mat')

        self.test_det_file = 'test.json'
        self.test_det_path = os.path.join(self.annot_path, self.test_det_file)

        self.mpii_joint_num = 16
        self.mpii_joints_name = (
        'R_Ankle', 'R_Knee', 'R_Hip', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Thorax', 'Neck', 'Head_top', 'R_Wrist',
        'R_Elbow', 'R_Shoulder', 'L_Shoulder', 'L_Elbow', 'L_Wrist')
        self.mpii_flip_pairs = ((0, 5), (1, 4), (2, 3), (10, 15), (11, 14), (12, 13))
        self.mpii_skeleton = (
        (0, 1), (1, 2), (2, 6), (3, 6), (3, 4), (4, 5), (6, 7), (7, 8), (8, 9), (10, 11), (11, 12), (7, 12), (7, 13),
        (13, 14), (14, 15))
        self.kps_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 1.2, 1.2, 1.2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.parent_ids = [1, 2, 6, 6, 3, 4, 6, 6, 7, 8, 11, 12, 7, 7, 13, 14]
        self.upper_body_ids = (7, 8, 9, 10, 11, 12, 13, 14, 15)
        self.lower_body_ids = (0, 1, 2, 3, 4, 5, 6)
        self.joint_num = self.mpii_joint_num
        self.image_size = np.array(cfg.input_img_shape)
        self.heatmap_size = np.array(cfg.output_hm_shape)
        self.sigma = cfg.sigma
        self.joints_weight = 1
        self.scale_factor = cfg.scale_factor
        self.rot_factor = cfg.rot_factor
        self.flip = cfg.flip
        self.target_type = cfg.target_type
        self.use_target_weight = cfg.use_target_weight
        self.use_different_joints_weight = cfg.use_different_joints_weight
        self.num_joints_half_body = 8
        self.prob_half_body = -1.0
        self.datalist = self.load_data()
        if data_split == 'train' and cfg.select_data:
            self.datalist = self.select_data(self.datalist)
        print("mpii data len: ", len(self.datalist))

    def load_data(self):
        data = list()

        if self.data_split == 'train':
            mpii = json.load(open(self.train_gt_path))
        elif self.data_split == 'val':
            mpii = json.load(open(self.val_gt_path))
        else:
            mpii = json.load(open(self.test_det_path))

        for d in mpii:
            img_name = d['image']

            center = np.array(d['center'], dtype=np.float32)
            scale = np.array([d['scale'], d['scale']], dtype=np.float32)

            if center[0] != -1:
                center[1] = center[1] + 15 * scale[1]
                scale = scale * 1.25
            center -= 1

            joints_3d = np.zeros((self.joint_num, 3), dtype=np.float32)
            joints_3d_vis = np.zeros((self.joint_num, 3), dtype=np.float32)

            if self.data_split != 'test':
                joints = np.array(d['joints'], dtype=np.float32)
                joints[:, 0:2] = joints[:, 0:2] - 1
                joints_vis = np.array(d['joints_vis'], dtype=np.float32)
                assert len(joints) == self.joint_num, 'joint num diff: {} vs {}'.format(len(joints), self.joint_num)

                joints_3d[:, 0:2] = joints[:, 0:2]
                joints_3d_vis[:, 0] = joints_vis[:]
                joints_3d_vis[:, 1] = joints_vis[:]

            data.append(
                {
                    'image': os.path.join(self.img_path, img_name),
                    'center': center,
                    'scale': scale,
                    'joints_3d': joints_3d,
                    'joints_3d_vis': joints_3d_vis,
                    'filename': '',
                    'imgnum': 0,
                }
            )

        return data

    def __len__(self):
        return len(self.datalist)

    def half_body_transform(self, joints, joints_vis):
        upper_joints = []
        lower_joints = []
        for joint_id in range(self.num_joints):
            if joints_vis[joint_id][0] > 0:
                if joint_id in self.upper_body_ids:
                    upper_joints.append(joints[joint_id])
                else:
                    lower_joints.append(joints[joint_id])

        if np.random.randn() < 0.5 and len(upper_joints) > 2:
            selected_joints = upper_joints
        else:
            selected_joints = lower_joints \
                if len(lower_joints) > 2 else upper_joints

        if len(selected_joints) < 2:
            return None, None

        selected_joints = np.array(selected_joints, dtype=np.float32)
        center = selected_joints.mean(axis=0)[:2]

        left_top = np.amin(selected_joints, axis=0)
        right_bottom = np.amax(selected_joints, axis=0)

        w = right_bottom[0] - left_top[0]
        h = right_bottom[1] - left_top[1]

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio

        scale = np.array(
            [
                w * 1.0 / self.pixel_std,
                h * 1.0 / self.pixel_std
            ],
            dtype=np.float32
        )

        scale = scale * 1.5

        return center, scale

    def select_data(self, datalist):
        datalist_selected = []
        for rec in datalist:
            num_vis = 0
            joints_x = 0.0
            joints_y = 0.0
            for joint, joint_vis in zip(
                    rec['joints_3d'], rec['joints_3d_vis']):
                if joint_vis[0] <= 0:
                    continue
                num_vis += 1

                joints_x += joint[0]
                joints_y += joint[1]
            if num_vis == 0:
                continue

            joints_x, joints_y = joints_x / num_vis, joints_y / num_vis

            area = rec['scale'][0] * rec['scale'][1] * (self.pixel_std ** 2)
            joints_center = np.array([joints_x, joints_y])
            bbox_center = np.array(rec['center'])
            diff_norm2 = np.linalg.norm((joints_center - bbox_center), 2)
            ks = np.exp(-1.0 * (diff_norm2 ** 2) / ((0.2) ** 2 * 2.0 * area))

            metric = (0.2 / 16) * num_vis + 0.45 - 0.2 / 16
            if ks > metric:
                datalist_selected.append(rec)

        logger.info('=> num db: {}'.format(len(datalist)))
        logger.info('=> num selected db: {}'.format(len(datalist_selected)))
        return datalist_selected

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])

        img_path = data['image']
        filename = data['filename'] if 'filename' in data else ''
        imgnum = data['imgnum'] if 'imgnum' in data else ''

        img = load_img(img_path)

        mpii_joint_img = data['joints_3d']
        mpii_joint_valid = data['joints_3d_vis']

        center = data['center']
        scale = data['scale']
        score = data['score'] if 'score' in data else 1
        rot = 0

        if self.data_split == 'train':
            if (np.sum(mpii_joint_valid[:, 0]) > self.num_joints_half_body and np.random.rand() < self.prob_half_body):
                c_half_body, s_half_body = self.half_body_transform(mpii_joint_img, mpii_joint_valid)
                if c_half_body is not None and s_half_body is not None:
                    center, scale = c_half_body, s_half_body

            sf = self.scale_factor
            rf = self.rot_factor
            scale = scale * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            rot = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) if random.random() <= 0.6 else 0

            if self.flip and random.random() <= 0.5:
                img = img[:, ::-1, :]
                mpii_joint_img[:, 0] = img.shape[1] - 1 - mpii_joint_img[:, 0]
                for pair in self.mpii_flip_pairs:
                    mpii_joint_img[pair[0], :], mpii_joint_img[pair[1], :] = mpii_joint_img[pair[1],
                                                                             :].copy(), mpii_joint_img[pair[0],
                                                                                        :].copy()
                    mpii_joint_valid[pair[0], :], mpii_joint_valid[pair[1], :] = mpii_joint_valid[pair[1],
                                                                                 :].copy(), mpii_joint_valid[pair[0],
                                                                                            :].copy()
                mpii_joint_img *= mpii_joint_valid
                center[0] = img.shape[1] - center[0] - 1

        trans = get_affine_transform_by_center(center, scale, rot, self.image_size)

        img = cv2.warpAffine(
            img,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)

        if self.transform:
            img = self.transform(img.astype(np.float32)) / 255.

        for i in range(self.joint_num):
            if mpii_joint_valid[i, 0] > 0.0:
                mpii_joint_img[i, 0:2] = affine_transform(mpii_joint_img[i, 0:2], trans)

        target, target_weight = self.generate_target(mpii_joint_img, mpii_joint_valid)

        heatmap = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)

        inputs = {'img': img}
        targets = {'heatmap': heatmap, 'target_weight': target_weight}
        meta_info = {
            'image': img_path,
            'filename': filename,
            'imgnum': imgnum,
            'joints': mpii_joint_img,
            'joints_vis': mpii_joint_valid,
            'center': center,
            'scale': scale,
            'rotation': rot,
            'score': score
        }

        return inputs, targets, meta_info

    def generate_target(self, joints, joints_vis):
        target_weight = np.ones((self.joint_num, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros((self.joint_num,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = self.sigma * 3

            for joint_id in range(self.joint_num):
                feat_stride = self.image_size / self.heatmap_size
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    target_weight[joint_id] = 0
                    continue

                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target, target_weight

    def evaluate(self, preds, output_dir, epoch, *args, **kwargs):

        preds = preds[:, :, 0:2] + 1.0

        SC_BIAS = 0.6
        threshold = 0.5

        if output_dir:
            pred_file = os.path.join(output_dir, f'pred_{epoch}.mat')
            savemat(pred_file, mdict={'preds': preds})

        if self.data_split == 'test':
            return {'Null': 0.0}, 0.0

        gt_file = os.path.join(self.val_gt_mat)
        gt_dict = loadmat(gt_file)
        dataset_joints = gt_dict['dataset_joints']
        jnt_missing = gt_dict['jnt_missing']
        pos_gt_src = gt_dict['pos_gt_src']
        headboxes_src = gt_dict['headboxes_src']

        pos_pred_src = np.transpose(preds, [1, 2, 0])

        head = np.where(dataset_joints == 'head')[1][0]
        lsho = np.where(dataset_joints == 'lsho')[1][0]
        lelb = np.where(dataset_joints == 'lelb')[1][0]
        lwri = np.where(dataset_joints == 'lwri')[1][0]
        lhip = np.where(dataset_joints == 'lhip')[1][0]
        lkne = np.where(dataset_joints == 'lkne')[1][0]
        lank = np.where(dataset_joints == 'lank')[1][0]

        rsho = np.where(dataset_joints == 'rsho')[1][0]
        relb = np.where(dataset_joints == 'relb')[1][0]
        rwri = np.where(dataset_joints == 'rwri')[1][0]
        rkne = np.where(dataset_joints == 'rkne')[1][0]
        rank = np.where(dataset_joints == 'rank')[1][0]
        rhip = np.where(dataset_joints == 'rhip')[1][0]

        jnt_visible = 1 - jnt_missing
        uv_error = pos_pred_src - pos_gt_src
        uv_err = np.linalg.norm(uv_error, axis=1)
        headsizes = headboxes_src[1, :, :] - headboxes_src[0, :, :]
        headsizes = np.linalg.norm(headsizes, axis=0)
        headsizes *= SC_BIAS
        scale = np.multiply(headsizes, np.ones((len(uv_err), 1)))
        scaled_uv_err = np.divide(uv_err, scale)
        scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible)
        jnt_count = np.sum(jnt_visible, axis=1)
        less_than_threshold = np.multiply((scaled_uv_err <= threshold),
                                          jnt_visible)
        PCKh = np.divide(100. * np.sum(less_than_threshold, axis=1), jnt_count)

        rng = np.arange(0, 0.5 + 0.01, 0.01)
        pckAll = np.zeros((len(rng), 16))

        for r in range(len(rng)):
            threshold = rng[r]
            less_than_threshold = np.multiply(scaled_uv_err <= threshold,
                                              jnt_visible)
            pckAll[r, :] = np.divide(100. * np.sum(less_than_threshold, axis=1),
                                     jnt_count)

        PCKh = np.ma.array(PCKh, mask=False)
        PCKh.mask[6:8] = True

        jnt_count = np.ma.array(jnt_count, mask=False)
        jnt_count.mask[6:8] = True
        jnt_ratio = jnt_count / np.sum(jnt_count).astype(np.float64)

        name_value = [
            ('Head', PCKh[head]),
            ('Shoulder', 0.5 * (PCKh[lsho] + PCKh[rsho])),
            ('Elbow', 0.5 * (PCKh[lelb] + PCKh[relb])),
            ('Wrist', 0.5 * (PCKh[lwri] + PCKh[rwri])),
            ('Hip', 0.5 * (PCKh[lhip] + PCKh[rhip])),
            ('Knee', 0.5 * (PCKh[lkne] + PCKh[rkne])),
            ('Ankle', 0.5 * (PCKh[lank] + PCKh[rank])),
            ('Mean', np.sum(PCKh * jnt_ratio)),
            ('Mean@0.1', np.sum(pckAll[11, :] * jnt_ratio))
        ]
        name_value = OrderedDict(name_value)

        print(name_value)
        return name_value, name_value['Mean']
