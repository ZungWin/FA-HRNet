import os
import os.path as osp
from collections import OrderedDict, defaultdict
import numpy as np
from config import cfg
import copy
import json_tricks as json
import cv2
import random
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import logging
logger = logging.getLogger(__name__)

from utils.preprocessing import load_img, get_affine_transform_by_center, affine_transform, soft_oks_nms, oks_nms

class MSCOCO(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split
        self.img_path = osp.join('..', 'data', 'MSCOCO', 'images')
        self.annot_path = osp.join('..', 'data', 'MSCOCO', 'annotations')
        self.rootnet_output_path = osp.join('..', 'data', 'MSCOCO', 'rootnet_output', 'bbox_root_coco_output.json')
        self.fitting_thr = 3.0
        self.nms_thre = cfg.nms_thre
        self.image_thre = cfg.image_thre
        self.soft_nms = cfg.soft_nms
        self.oks_thre = cfg.oks_thre
        self.in_vis_thre = cfg.in_vis_thre
        self.image_set = cfg.train_set if data_split == 'train' else cfg.val_set
        self.image_set = cfg.test_set if data_split == 'test' else self.image_set
        self.coco_joint_num = 17
        self.coco_joints_name = (
            'Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist',
            'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle')
        self.coco_skeleton = (
            (1, 2), (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (5, 7), (7, 9), (12, 14), (14, 16), (11, 13),
            (13, 15),
            (5, 6), (11, 12))
        self.coco_flip_pairs = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16))
        self.coco_root_joint_idx = 16
        self.parent_ids = None
        self.upper_body_ids = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        self.lower_body_ids = (11, 12, 13, 14, 15, 16)
        self.joint_num = self.coco_joint_num
        self.sigma = cfg.sigma
        self.joints_weight = np.array(
            [
                1., 1., 1., 1., 1., 1., 1., 1.2, 1.2,
                1.5, 1.5, 1., 1., 1.2, 1.2, 1.5, 1.5
            ],
            dtype=np.float32
        ).reshape((self.joint_num, 1))
        self.coco = COCO(self._get_ann_file_keypoint())
        cats = [cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.classes = ['__background__'] + cats
        logger.info('=> classes: {}'.format(self.classes))
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        self._coco_ind_to_class_ind = dict(
            [
                (self._class_to_coco_ind[cls], self._class_to_ind[cls])
                for cls in self.classes[1:]
            ]
        )
        # load image file names
        self.image_set_index = self._load_image_set_index()
        self.num_images = len(self.image_set_index)
        logger.info('=> num_images: {}'.format(self.num_images))
        self.num_joints_half_body = cfg.num_joints_half_body
        self.prob_half_body = cfg.prob_half_body
        self.use_gt_bbox = cfg.use_gt_bbox
        self.image_size = np.array(cfg.input_img_shape)
        self.heatmap_size = np.array(cfg.output_hm_shape)
        self.target_type = cfg.target_type
        self.image_width = cfg.input_img_shape[0]
        self.image_height = cfg.input_img_shape[1]
        self.aspect_ratio = self.image_width * 1.0 / self.image_height
        self.scale_factor = cfg.scale_factor
        self.rot_factor = cfg.rot_factor
        self.flip = cfg.flip
        self.data_format = cfg.data_format
        self.use_different_joints_weight = cfg.use_different_joints_weight
        self.pixel_std = 200
        self.bbox_file = osp.join(self.annot_path,
                                  'COCO_val2017_detections_AP_H_56_person.json') if self.data_split == 'val' else osp.join(
            self.annot_path, 'COCO_test-dev2017_detections_AP_H_609_person.json')
        self.datalist = self.load_data()
        if self.data_split == 'train' and cfg.select_data:
            self.datalist = self.select_data(self.datalist)

        print("coco data len: ", len(self.datalist))

    def _get_ann_file_keypoint(self):
        """ self.root / annotations / person_keypoints_train2017.json """
        prefix = 'person_keypoints' \
            if 'test' not in self.image_set else 'image_info'
        return os.path.join(
            self.annot_path,
            prefix + '_' + self.image_set + '.json'
        )

    def _load_image_set_index(self):
        """ image id: int """
        image_ids = self.coco.getImgIds()
        return image_ids

    def load_data(self):
        if self.data_split == 'train' or self.use_gt_bbox:
            # use ground truth bbox
            datalist = self._load_coco_keypoint_annotations()
        else:
            # use bbox from detection
            datalist = self._load_coco_person_detection_results()
        return datalist

    def _load_coco_keypoint_annotations(self):
        """ ground truth bbox and keypoints """
        datalist = []
        for index in self.image_set_index:
            datalist.extend(self._load_coco_keypoint_annotation_kernal(index))
        return datalist

    def _load_coco_keypoint_annotation_kernal(self, index):
        """
        coco ann: [u'segmentation', u'area', u'iscrowd', u'image_id', u'bbox', u'category_id', u'id']
        iscrowd:
            crowd instances are handled by marking their overlaps with all categories to -1
            and later excluded in training
        bbox:
            [x1, y1, w, h]
        :param index: coco image id
        :return: db entry
        """
        im_ann = self.coco.loadImgs(index)[0]
        width = im_ann['width']
        height = im_ann['height']

        annIds = self.coco.getAnnIds(imgIds=index, iscrowd=False)
        objs = self.coco.loadAnns(annIds)

        # sanitize bboxes
        valid_objs = []
        for obj in objs:
            x, y, w, h = obj['bbox']
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2 - x1, y2 - y1]
                valid_objs.append(obj)
        objs = valid_objs

        rec = []
        for obj in objs:
            cls = self._coco_ind_to_class_ind[obj['category_id']]
            if cls != 1:
                continue

            # ignore objs without keypoints annotation
            if max(obj['keypoints']) == 0:
                continue

            joints_3d = np.zeros((self.joint_num, 3), dtype=np.float)
            joints_3d_vis = np.zeros((self.joint_num, 3), dtype=np.float)
            for ipt in range(self.joint_num):
                joints_3d[ipt, 0] = obj['keypoints'][ipt * 3 + 0]
                joints_3d[ipt, 1] = obj['keypoints'][ipt * 3 + 1]
                joints_3d[ipt, 2] = 0
                t_vis = obj['keypoints'][ipt * 3 + 2]
                if t_vis > 1:
                    t_vis = 1
                joints_3d_vis[ipt, 0] = t_vis
                joints_3d_vis[ipt, 1] = t_vis
                joints_3d_vis[ipt, 2] = 0

            center, scale = self._box2cs(obj['clean_bbox'][:4])
            rec.append({
                'image': self.image_path_from_index(index),
                'center': center,
                'scale': scale,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
                'filename': '',
                'imgnum': 0,
            })

        return rec

    def _load_coco_person_detection_results(self):
        all_boxes = None
        with open(self.bbox_file, 'r') as f:
            all_boxes = json.load(f)

        if not all_boxes:
            logger.error('=> Load %s fail!' % self.bbox_file)
            return None

        logger.info('=> Total boxes: {}'.format(len(all_boxes)))

        kpt_db = []
        num_boxes = 0
        for n_img in range(0, len(all_boxes)):
            det_res = all_boxes[n_img]
            if det_res['category_id'] != 1:
                continue
            img_name = self.image_path_from_index(det_res['image_id'])
            box = det_res['bbox']
            score = det_res['score']

            if score < self.image_thre:
                continue

            num_boxes = num_boxes + 1

            center, scale = self._box2cs(box)
            joints_3d = np.zeros((self.joint_num, 3), dtype=np.float64)
            joints_3d_vis = np.ones(
                (self.joint_num, 3), dtype=np.float64)
            kpt_db.append({
                'image': img_name,
                'center': center,
                'scale': scale,
                'score': score,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
            })

        logger.info('=> Total boxes after fliter low score@{}: {}'.format(
            self.image_thre, num_boxes))
        return kpt_db

    def image_path_from_index(self, index):
        """ example: images / train2017 / 000000119993.jpg """
        file_name = '%012d.jpg' % index
        if '2014' in self.image_set:
            file_name = 'COCO_%s_' % self.image_set + file_name

        prefix = 'test2017' if 'test' in self.image_set else self.image_set

        data_name = prefix + '.zip@' if self.data_format == 'zip' else prefix

        image_path = os.path.join(
            self.img_path, data_name, file_name)

        return image_path

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale

    def half_body_transform(self, joints, joints_vis):
        upper_joints = []
        lower_joints = []
        for joint_id in range(self.joint_num):
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

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_path = data['image']
        filename = data['filename'] if 'filename' in data else ''
        imgnum = data['imgnum'] if 'imgnum' in data else ''

        img = load_img(img_path)
        if img is None:
            logger.error('=> fail to read {}'.format(img_path))
            raise ValueError('Fail to read {}'.format(img_path))

        coco_joint_img = data['joints_3d']
        coco_joint_valid = data['joints_3d_vis']
        center = data['center']
        scale = data['scale']
        score = data['score'] if 'score' in data else 1
        rot = 0

        if self.data_split == 'train':
            # coco gt
            # transform coco joints to target db joints
            if (np.sum(coco_joint_valid[:, 0]) > self.num_joints_half_body and np.random.rand() < self.prob_half_body):
                c_half_body, s_half_body = self.half_body_transform(coco_joint_img, coco_joint_valid)
                if c_half_body is not None and s_half_body is not None:
                    center, scale = c_half_body, s_half_body

            sf = self.scale_factor
            rf = self.rot_factor
            scale = scale * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            rot = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) if random.random() <= 0.6 else 0

            if self.flip and random.random() <= 0.5:
                img = img[:, ::-1, :]
                coco_joint_img[:, 0] = img.shape[1] - 1 - coco_joint_img[:, 0]
                for pair in self.coco_flip_pairs:
                    coco_joint_img[pair[0], :], coco_joint_img[pair[1], :] = coco_joint_img[pair[1],
                                                                             :].copy(), coco_joint_img[pair[0],
                                                                                        :].copy()
                    coco_joint_valid[pair[0], :], coco_joint_valid[pair[1], :] = coco_joint_valid[pair[1],
                                                                                 :].copy(), coco_joint_valid[pair[0],
                                                                                            :].copy()
                coco_joint_img *= coco_joint_valid
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
            if coco_joint_valid[i, 0] > 0.0:
                coco_joint_img[i, 0:2] = affine_transform(coco_joint_img[i, 0:2], trans)

        target, target_weight = self.generate_target(coco_joint_img, coco_joint_valid)

        save_path = os.path.join(cfg.vis_dir, f'heatmap_{idx}.jpg')

        heatmap = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)

        inputs = {'img': img}
        targets = {'heatmap': heatmap, 'target_weight': target_weight}
        meta_info = {
            'image': img_path,
            'filename': filename,
            'imgnum': imgnum,
            'joints': coco_joint_img,
            'joints_vis': coco_joint_valid,
            'center': center,
            'scale': scale,
            'rotation': rot,
            'score': score
        }

        return inputs, targets, meta_info

    def generate_target(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
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
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target, target_weight

    def evaluate(self, cfg, preds, output_dir, all_boxes, img_path,
                 *args, **kwargs):
        rank = cfg.rank

        res_folder = os.path.join(output_dir, 'results')
        if not os.path.exists(res_folder):
            try:
                os.makedirs(res_folder)
            except Exception:
                logger.error('Fail to make {}'.format(res_folder))

        res_file = os.path.join(
            res_folder, 'keypoints_{}_results_{}.json'.format(
                self.image_set, rank)
        )

        # person x (keypoints)
        _kpts = []
        for idx, kpt in enumerate(preds):
            _kpts.append({
                'keypoints': kpt,
                'center': all_boxes[idx][0:2],
                'scale': all_boxes[idx][2:4],
                'area': all_boxes[idx][4],
                'score': all_boxes[idx][5],
                'image': int(img_path[idx][-16:-4])
            })
        # image x person x (keypoints)
        kpts = defaultdict(list)
        for kpt in _kpts:
            kpts[kpt['image']].append(kpt)

        # rescoring and oks nms
        num_joints = self.joint_num
        in_vis_thre = self.in_vis_thre
        oks_thre = self.oks_thre
        oks_nmsed_kpts = []
        for img in kpts.keys():
            img_kpts = kpts[img]
            for n_p in img_kpts:
                box_score = n_p['score']
                kpt_score = 0
                valid_num = 0
                for n_jt in range(0, num_joints):
                    t_s = n_p['keypoints'][n_jt][2]
                    if t_s > in_vis_thre:
                        kpt_score = kpt_score + t_s
                        valid_num = valid_num + 1
                if valid_num != 0:
                    kpt_score = kpt_score / valid_num
                # rescoring
                n_p['score'] = kpt_score * box_score

            if self.soft_nms:
                keep = soft_oks_nms(
                    [img_kpts[i] for i in range(len(img_kpts))],
                    oks_thre
                )
            else:
                keep = oks_nms(
                    [img_kpts[i] for i in range(len(img_kpts))],
                    oks_thre
                )

            if len(keep) == 0:
                oks_nmsed_kpts.append(img_kpts)
            else:
                oks_nmsed_kpts.append([img_kpts[_keep] for _keep in keep])

        self._write_coco_keypoint_results(
            oks_nmsed_kpts, res_file)
        if 'test' not in self.image_set:
            info_str = self._do_python_keypoint_eval(
                res_file, res_folder)
            name_value = OrderedDict(info_str)
            return name_value, name_value['AP']
        else:
            return {'Null': 0}, 0

    def _write_coco_keypoint_results(self, keypoints, res_file):
        data_pack = [
            {
                'cat_id': self._class_to_coco_ind[cls],
                'cls_ind': cls_ind,
                'cls': cls,
                'ann_type': 'keypoints',
                'keypoints': keypoints
            }
            for cls_ind, cls in enumerate(self.classes) if not cls == '__background__'
        ]

        results = self._coco_keypoint_results_one_category_kernel(data_pack[0])
        logger.info('=> writing results json to %s' % res_file)
        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)
        try:
            json.load(open(res_file))
        except Exception:
            content = []
            with open(res_file, 'r') as f:
                for line in f:
                    content.append(line)
            content[-1] = ']'
            with open(res_file, 'w') as f:
                for c in content:
                    f.write(c)

    def _coco_keypoint_results_one_category_kernel(self, data_pack):
        cat_id = data_pack['cat_id']
        keypoints = data_pack['keypoints']
        cat_results = []

        for img_kpts in keypoints:
            if len(img_kpts) == 0:
                continue

            _key_points = np.array([img_kpts[k]['keypoints']
                                    for k in range(len(img_kpts))])
            key_points = np.zeros(
                (_key_points.shape[0], self.joint_num * 3), dtype=np.float64
            )

            for ipt in range(self.joint_num):
                key_points[:, ipt * 3 + 0] = _key_points[:, ipt, 0]
                key_points[:, ipt * 3 + 1] = _key_points[:, ipt, 1]
                key_points[:, ipt * 3 + 2] = _key_points[:, ipt, 2]  # keypoints score.

            result = [
                {
                    'image_id': img_kpts[k]['image'],
                    'category_id': cat_id,
                    'keypoints': list(key_points[k]),
                    'score': img_kpts[k]['score'],
                    'center': list(img_kpts[k]['center']),
                    'scale': list(img_kpts[k]['scale'])
                }
                for k in range(len(img_kpts))
            ]
            cat_results.extend(result)

        return cat_results

    def _do_python_keypoint_eval(self, res_file, res_folder):
        coco_dt = self.coco.loadRes(res_file)
        coco_eval = COCOeval(self.coco, coco_dt, 'keypoints')
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats_names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']

        info_str = []
        for ind, name in enumerate(stats_names):
            info_str.append((name, coco_eval.stats[ind]))

        return info_str

