from __future__ import print_function

import argparse
import os
import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import time
from dsgn.models import *
from dsgn.utils.numpy_utils import *
from dsgn.utils.numba_utils import *
from dsgn.utils.torch_utils import *
from env_utils import *
from dsgn.models.inference3d import make_fcos3d_postprocessor

parser = argparse.ArgumentParser(description='DSGN 3D Detection Inference Only')
parser.add_argument('-cfg', '--cfg', '--config', default=None, help='config path')
parser.add_argument('--data_path', default=None, help='data path')#'./data/kitti/training'
parser.add_argument('--loadmodel', default=None, help='loading model')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--split_file', default=None, help='split file')
parser.add_argument('--save_path', type=str, default='./outputs/result')
parser.add_argument('--btest', type=int, default=None)
parser.add_argument('--devices', type=str, default=None)
parser.add_argument('--tag', type=str, default='')
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--debugnum', default=None, type=int)
parser.add_argument('--train', action='store_true', default=False)
args = parser.parse_args()

if not args.devices:
    args.devices = str(np.argmin(mem_info()))

if args.devices is not None and '-' in args.devices:
    gpus = args.devices.split('-')
    gpus[0] = 0 if not gpus[0].isdigit() else int(gpus[0])
    gpus[1] = len(mem_info()) if not gpus[1].isdigit() else int(gpus[1]) + 1
    args.devices = ','.join(map(lambda x: str(x), list(range(*gpus))))

if args.debugnum is None:
    args.debugnum = 100

exp = Experimenter(os.path.dirname(args.loadmodel), args.cfg)
cfg = exp.config

if args.debug:
    args.btest = len(args.devices.split(','))
    num_workers = 0
    cfg.debug = True
    args.tag += 'debug{}'.format(args.debugnum)
else:
    num_workers = 12

if args.train:
    args.split_file = './data/kitti/train.txt'
    args.tag += '_train'

assert args.btest

print('Using GPU:{}'.format(args.devices))
os.environ['CUDA_VISIBLE_DEVICES'] = args.devices

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

from dsgn.dataloader import KITTILoader3D as ls
from dsgn.dataloader import KITTILoader_dataset3d as DA

# Only load image paths and calibration, skip disparity/depth ground-truth
all_left_img, all_right_img, all_left_disp = ls.dataloader(
    args.data_path,
    args.split_file,
    depth_disp=False,
    cfg=cfg,
    is_train=False
)

class BatchCollator(object):
    def __call__(self, batch):
        transpose_batch = list(zip(*batch))
        l = torch.cat(transpose_batch[0], dim=0)
        r = torch.cat(transpose_batch[1], dim=0)
        calib = transpose_batch[3]
        calib_R = transpose_batch[4]
        image_sizes = transpose_batch[5]
        image_indexes = transpose_batch[6]
        outputs = [l, r, calib, calib_R, image_sizes, image_indexes]
        return outputs

# Note: myImageFloder should support missing disparity/GT (pass None or skip if needed).
ImageFloader = DA.myImageFloder(
    all_left_img, all_right_img, all_left_disp, False, split=args.split_file, cfg=cfg
)

TestImgLoader = torch.utils.data.DataLoader(
    ImageFloader,
    batch_size=args.btest, shuffle=False, num_workers=num_workers, drop_last=False,
    collate_fn=BatchCollator()
)

model = StereoNet(cfg=cfg)
model = nn.DataParallel(model)
model.cuda()

if args.loadmodel is not None and args.loadmodel.endswith('tar'):
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'], strict=False)
    print('Loaded {}'.format(args.loadmodel))
else:
    print('------------------------------ Load Nothing ---------------------------------')

print('Number of model parameters: {}'.format(
    sum([p.data.nelement() for p in model.parameters()])))

def test(imgL, imgR, image_sizes=None, calibs_fu=None, calibs_baseline=None, calibs_Proj=None, calibs_Proj_R=None):
    model.eval()
    with torch.no_grad():
        outputs = model(imgL, imgR, calibs_fu, calibs_baseline,
                        calibs_Proj, calibs_Proj_R=calibs_Proj_R)
    pred_disp = outputs['depth_preds']
    rets = [pred_disp]
    if cfg.RPN3D_ENABLE:
        box_pred = make_fcos3d_postprocessor(cfg)(
            outputs['bbox_cls'], outputs['bbox_reg'], outputs['bbox_centerness'],
            image_sizes=image_sizes, calibs_Proj=calibs_Proj)
        rets.append(box_pred)
    return rets

def kitti_output(box_pred_left, image_indexes, output_path, box_pred_right=None):
    for i, (prediction, image_index) in enumerate(zip(box_pred_left, image_indexes)):
        with open(os.path.join(output_path, '{:06d}.txt'.format(image_index)), 'w') as f:
            for i, (cls, bbox, score) in enumerate(zip(
                    prediction.get_field('labels').cpu(), prediction.bbox.cpu(),
                    prediction.get_field('scores').cpu())):
                if prediction.has_field('box_corner3d'):
                    assert cls != 0
                    box_corner3d = prediction.get_field(
                        'box_corner3d').cpu()[i].reshape(8, 3)
                    box_center3d = box_corner3d.mean(dim=0)
                    x, y, z = box_center3d
                    box_corner3d = box_corner3d - box_center3d.view(1, 3)
                    h, w, l, ry = get_dimensions(box_corner3d.transpose(0, 1))
                    if getattr(cfg, 'learn_viewpoint', False):
                        ry = ry - np.arctan2(z, x) + np.pi / 2
                else:
                    h, w, l = 0., 0., 0.
                    box_center3d = [0., 0., 0.]
                    ry = 0.
                cls_type = 'Pedestrian' if cls == 1 else 'Car' if cls == 2 else 'Cyclist'
                f.write('{} -1 -1 {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.8f}\n'.format(
                    cls_type,
                    -np.arctan2(box_center3d[0], box_center3d[2]) + ry,
                    bbox[0], bbox[1],
                    bbox[2], bbox[3],
                    h, w, l,
                    box_center3d[0], box_center3d[1] + h / 2., box_center3d[2],
                    ry,
                    score))
        print('Wrote {}'.format(image_index))

def main():
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    output_path = os.path.join(os.path.dirname(os.path.abspath(
        __file__)), '..', os.path.dirname(args.loadmodel))
    kitti_out_dir = output_path + '/awsim_output_2' + args.tag
    if not os.path.exists(kitti_out_dir):
        os.makedirs(kitti_out_dir)
    else:
        os.system('rm -rf {}/*'.format(kitti_out_dir))

    for batch_idx, databatch in enumerate(TestImgLoader):
        imgL, imgR, calib_batch, calib_R_batch, image_sizes, image_indexes = databatch

        if cfg.debug:
            if batch_idx * len(imgL) > args.debugnum:
                break

        imgL, imgR = imgL.cuda(), imgR.cuda()

        calibs_fu = torch.as_tensor([c.f_u for c in calib_batch])
        calibs_baseline = torch.as_tensor(
            [(c.P[0, 3] - c_R.P[0, 3]) / c.P[0, 0] for c, c_R in zip(calib_batch, calib_R_batch)])
        calibs_Proj = torch.as_tensor([c.P for c in calib_batch])
        calibs_Proj_R = torch.as_tensor([c.P for c in calib_R_batch])

        start_time = time.time()
        output = test(imgL, imgR, image_sizes=image_sizes, calibs_fu=calibs_fu,
                      calibs_baseline=calibs_baseline, calibs_Proj=calibs_Proj, calibs_Proj_R=calibs_Proj_R)
        if cfg.RPN3D_ENABLE:
            pred_disp, box_pred = output
            kitti_output(box_pred[0], image_indexes, kitti_out_dir)
        else:
            pred_disp, = output
        print('time = %.2f' % (time.time() - start_time))

if __name__ == '__main__':
    main()