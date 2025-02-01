import os
# os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import torch
import argparse
from tqdm import tqdm
import numpy as np
import torch.backends.cudnn as cudnn
from config import cfg
from base import Tester
from utils.inference import get_final_preds
import logging
import pdb

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--test_epoch', type=str, dest='test_epoch')
    parser.add_argument('--exp_dir', type=str, default='')
    parser.add_argument('--cfg', type=str, default='', help='experiment configure file name')

    args = parser.parse_args()

    if not args.gpu_ids:
        assert 0, "Please set propoer gpu ids"

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    assert args.test_epoch, 'Test epoch is required.'
    return args


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values + 1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
        ' |'
    )


def main():
    args = parse_args()
    cfg.set_args(args.gpu_ids, is_test=True, exp_dir=args.exp_dir)
    cudnn.benchmark = True
    if args.cfg:
        cfg.update(args.cfg)

    tester = Tester(args.test_epoch)
    tester._make_batch_generator()
    tester._make_model()

    eval_result = {}
    cur_sample_idx = 0

    all_preds = np.zeros((tester.num_samples, tester.joint_num, 3), dtype=np.float32)
    all_boxes = np.zeros((tester.num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0

    for itr, (inputs, targets, meta_info) in enumerate(tqdm(tester.batch_generator)):
        # forward
        with torch.no_grad():
            out = tester.model(inputs, targets, meta_info)

        c = meta_info['center'].numpy()
        s = meta_info['scale'].numpy()
        score = meta_info['score'].numpy()
        num_images = inputs['img'].size(0)

        # pdb.set_trace()

        preds, maxvals = get_final_preds(out.clone().cpu().numpy(), c, s)

        all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
        all_preds[idx:idx + num_images, :, 2:3] = maxvals
        # double check this all_boxes parts
        all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
        all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
        all_boxes[idx:idx + num_images, 4] = np.prod(s * 200, 1)
        all_boxes[idx:idx + num_images, 5] = score
        image_path.extend(meta_info['image'])
        idx += num_images

    output_dir = cfg.result_dir

    # evaluate
    name_values, perf_indicator = tester.testset.evaluate(
        cfg, all_preds, output_dir, all_boxes, image_path,
        filenames, imgnums
    )
    model_name = 'FA-HRNet'
    if isinstance(name_values, list):
        for name_value in name_values:
            _print_name_value(name_value, model_name)
    else:
        _print_name_value(name_values, model_name)


if __name__ == "__main__":
    main()
