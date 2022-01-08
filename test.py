from __future__ import print_function
import argparse, json
import os
from utils.data_utils import imagesc
import torch
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from utils.data_utils import norm_01

load_dotenv('.env')


def get_model(epochs, name, dir_checkpoints, device, eval=False):
    model_path = dir_checkpoints + "{}/{}_model_epoch_{}.pth".format(args.prj, name, epochs)
    net = torch.load(model_path).to(device)
    if eval:
        net.eval()
    else:
        net.train()
    return net


def overlap_red(x0, y0):
    y = 1 * y0
    x = 1 * x0
    x = x - x.min()
    x = x / x.max()

    c = 0
    x[1, y == c] = 0.1 * x[1, y == c]
    x[2, y == c] = 0.1 * x[2, y == c]

    c = 1
    x[0, y == c] = 0.1 * x[0, y == c]
    x[2, y == c] = 0.1 * x[2, y == c]

    c = 2
    x[0, y == c] = 0.1 * x[0, y == c]
    x[1, y == c] = 0.1 * x[1, y == c]
    return x


class Pix2PixModel:
    def __init__(self, args):
        self.args = args
        self.dir_checkpoints = os.environ.get('CHECKPOINTS')
        from dataloader.data import DatasetFromFolder as Dataset

        if args.testset:
            self.test_set = Dataset(os.environ.get('DATASET') + args.testset + '/test/', args, mode='test', unpaired=False)
        else:
            self.test_set = Dataset(os.environ.get('DATASET') + args.dataset + '/test/', args, mode='test', unpaired=False)

        os.makedirs(os.path.join("outputs/results", args.prj), exist_ok=True)

        self.seg_model = torch.load(os.environ.get('model_seg')).cuda()
        self.netg_t2d = torch.load(os.environ.get('model_t2d')).cuda()

        self.device = torch.device("cuda:0")

    def get_one_output(self, i, xy, alpha=None):
        x = self.test_set.__getitem__(i)

        # model
        oriX = x[0].unsqueeze(0).to(self.device)
        oriY = x[1].unsqueeze(0).to(self.device)

        if xy == 'x':
            in_img = oriX
            out_img = oriY
        elif xy == 'y':
            in_img = oriY
            out_img = oriX

        alpha = alpha / 100

        try:
            output = self.net_g(in_img, alpha * torch.ones(1, 2).cuda())
        except:
            try:
                output = self.net_g(in_img, alpha * torch.ones(1, 1).cuda())
            except:
                output = self.net_g(in_img)

        in_img = in_img.detach().cpu()[0, ::]
        out_img = out_img.detach().cpu()[0, ::]
        output = output[0][0, ::].detach().cpu()

        return in_img, out_img, output

    def get_t2d(self, ori):
        t2d = self.netg_t2d(ori.cuda().unsqueeze(0))[0][0, ::].detach().cpu()
        return t2d

    def get_seg(self, ori):
        seg = self.seg_model(ori.cuda().unsqueeze(0))
        seg = torch.argmax(seg, 1)[0,::].detach().cpu()
        return seg

    def get_all_seg(self, input, t2d=True):
        if t2d:
            input = list(map(lambda k: list(map(lambda v: self.get_t2d(v), k)), input))  # (3, 256, 256) (-1, 1)
        list_norm = list(map(lambda k: list(map(lambda v: norm_01(v), k)), input))  # (3, 256, 256) (0, 1)
        list_seg = list(map(lambda k: list(map(lambda v: self.get_seg(v), k)), list_norm))
        return list_seg


# Command Line Argument
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--jsn', type=str, default='default', help='name of ini file')
parser.add_argument('--dataset', help='name of training dataset')
parser.add_argument('--testset', help='name of testing dataset if different than the training dataset')
parser.add_argument('--prj', type=str, help='name of the project')
parser.add_argument('--direction', type=str, help='a2b or b2a')
parser.add_argument('--netg', type=str)
parser.add_argument('--resize', type=int)
parser.add_argument('--flip', action='store_true', dest='flip')
parser.add_argument('--eval', action='store_true', dest='eval')
parser.add_argument('--nepochs', default=(0, 10, 200), nargs='+', help='which checkpoints to be interfered with', type=int)
parser.add_argument('--nalpha', default=(0, 100, 2), nargs='+', help='list of alphas', type=int)
parser.add_argument('--mode', type=str, default='dummy')
parser.add_argument('--port', type=str, default='dummy')
with open('outputs/' + parser.parse_args().jsn + '.json', 'rt') as f:
    t_args = argparse.Namespace()
    t_args.__dict__.update(json.load(f))
    args = parser.parse_args(namespace=t_args)
args.prj = args.dataset + '_' + args.prj


test_unit = Pix2PixModel(args=args)
for epoch in range(*args.nepochs):
    test_unit.net_g = get_model(epoch, args.netg, test_unit.dir_checkpoints, test_unit.device, eval=args.eval)
    for alpha in np.linspace(*args.nalpha):
        out_xy = list(map(lambda v: test_unit.get_one_output(v, 'x', alpha), args.irange))
        out_yx = list(map(lambda v: test_unit.get_one_output(v, 'y', alpha), args.irange))

        out_xy = list(zip(*out_xy))
        out_yx = list(zip(*out_yx))

        seg_xy = test_unit.get_all_seg(out_xy)
        seg_yx = test_unit.get_all_seg(out_yx)

        diff_xy = [(x[1] - x[0]) for x in list(zip(out_xy[2], out_xy[0]))]
        diff_yx = [(x[1] - x[0]) for x in list(zip(out_yx[2], out_yx[0]))]

        diff_xy[0][0, 0, 0] = 2
        diff_xy[0][0, 0, 1] = -2
        diff_yx[0][0, 0, 0] = 2
        diff_yx[0][0, 0, 1] = -2

        to_show = [out_xy[0],
                   list(map(lambda x, y: overlap_red(x, y), out_xy[0], seg_xy[0])),
                   out_xy[1],
                   list(map(lambda x, y: overlap_red(x, y), out_xy[1], seg_xy[1])),
                   out_xy[2],
                   diff_xy,
                   list(map(lambda x, y: overlap_red(x, y), diff_xy, seg_xy[2])),
                   ]

        to_show = [torch.cat(x, len(x[0].shape) - 1) for x in to_show]
        to_show = [x - x.min() for x in to_show]

        for i in range(len(to_show)):
            if len(to_show[i].shape) == 2:
                to_show[i] = torch.cat([to_show[i].unsqueeze(0)]*3, 0)

        imagesc(np.concatenate([x / x.max() for x in to_show], 1), show=False,
                save=os.path.join("outputs/results", args.prj, str(epoch) + '_' + str(alpha) + '.jpg'))



# USAGE
# CUDA_VISIBLE_DEVICES=0 python test.py --jsn default --dataset pain --nalpha 0 100 2  --prj VryAtt
# CUDA_VISIBLE_DEVICES=0 python test.py --jsn default --dataset TSE_DESS --nalpha 0 100 1  --prj VryCycleUP --net netGXY