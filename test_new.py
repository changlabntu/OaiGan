
from __future__ import print_function
import argparse
import os
from utils.data_utils import imagesc
import torch
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
from utils.make_config import load_config
from dotenv import load_dotenv

load_dotenv('.env')


def norm_01(x):
    """
    normalize to 0 - 1
    """
    x = x - x.min()
    x = x / x.max()
    return x


def get_model(epochs, name, dir_checkpoints, device):
    model_path = dir_checkpoints + "{}/{}_model_epoch_{}.pth".format(opt.prj, name, epochs)
    net = torch.load(model_path).to(device)
    return net


def overlap_red(x0, y0):
    y = 1 * y0
    x = 1 * x0
    c = 0
    x[0, y == c] = 0.0 * x[0, y == c]
    c = 2
    x[1, y == c] = 0.0 * x[1, y == c]
    c = 1
    x[2, y == c] = 0.0 * x[2, y == c]
    return x


def overlap_red2(x0, y0, channel):
    y = 1 * y0
    x = 1 * x0
    c = 0
    x[0, y == c] = 1
    x[1, y == c] = 0.0 * x[1, y == c]
    x[2, y == c] = 0.0 * x[1, y == c]
    return x


# Testing settings
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--dataset', default='TSE_DESS', help='name of training dataset')
parser.add_argument('--testset', default='TSE_DESS', help='name of testing dataset if different than the training dataset')
parser.add_argument('--prj', type=str, default='NoResampleResnet9', help='name of the project')
parser.add_argument('--direction', type=str, default='a_b', help='a2b or b2a')
parser.add_argument('--flip', action='store_true', dest='flip', default=False, help='image flip left right')
parser.add_argument('--crop', type=int, default=0)
parser.add_argument('--nepochs', nargs='+', default=[0, 601, 20], help='which checkpoints to be interfered with')
parser.add_argument('--mode', type=str, default='dummy')
parser.add_argument('--port', type=str, default='dummy')
parser.add_argument('--att', action='store_true', dest='att', default=False)
parser.add_argument('--cycle', action='store_true', dest='cycle', default=False)
parser.add_argument('--t2d', action='store_true', dest='t2d', default=True)

opt = parser.parse_args()
opt.prj = opt.dataset + '_' + opt.prj
opt.nepochs = range(int(opt.nepochs[0]), int(opt.nepochs[1]), int(opt.nepochs[2]))
print(opt)


class Pix2PixModel:
    def __init__(self, opt):
        self.opt = opt
        self.dir_checkpoints = os.environ.get('CHECKPOINTS')
        from dataloader.data_no_resample import DatasetFromFolder as Dataset

        if opt.testset:
            self.test_set = Dataset(os.environ.get('DATASET') + opt.testset + '/test/', opt, mode='test')
        else:
            self.test_set = Dataset(os.environ.get('DATASET') + opt.dataset + '/test/', opt, mode='test')

        if not os.path.exists(os.path.join("result", opt.prj)):
            os.makedirs(os.path.join("result", opt.prj))

        self.seg_model = torch.load(os.environ.get('model_seg')).cuda()
        self.netg_t2d = torch.load(os.environ.get('model_t2d')).cuda()

        self.device = torch.device("cuda:0")

        self.irange = [15, 19, 34, 79, 95, 109, 172, 173, 208, 249, 266]  # pain
        # irange = [102, 109, 128, 190, 193, 235, 252, 276, 318, 335]

    def get_one_output(self, i):
        opt = self.opt
        x = self.test_set.__getitem__(i)
        img = x[0]
        mask = x[1]

        # model
        input = img.unsqueeze(0).to(self.device)
        if not opt.cycle:
            net_g = get_model(epoch, 'netG', self.dir_checkpoints, self.device)
        else:
            net_g = get_model(epoch, 'netG_ab', self.dir_checkpoints, self.device)
        output = net_g(input)
        out = output[0][0, ::].detach().cpu()
        if self.opt.att:
            return img, mask, out, output[1][0, ::].detach().cpu()
        else:
            return img, mask, out

    def get_t2d(self, ori):
        t2d = self.netg_t2d(ori.cuda().unsqueeze(0))[0][0, ::].detach().cpu()
        return t2d

    def get_seg(self, ori):
        seg = self.seg_model(ori.cuda().unsqueeze(0))[0]
        seg = torch.argmax(seg, 1)[0,::].detach().cpu()
        return seg


test_unit = Pix2PixModel(opt=opt)
irange = list(np.random.randint(0, 250, 10))#[15, 19, 34, 79, 95, 109, 172, 173, 208, 249, 266]

for epoch in opt.nepochs:
    outputs = list(map(lambda v: test_unit.get_one_output(v), irange))  # (3, 256, 256) (-1, 1)
    list_ori = list(zip(*outputs))  # (3, 256, 256) (-1, 1)
    list_ori_norm = list(map(lambda k: list(map(lambda v: norm_01(v), k)), list_ori[:3]))  # (3, 256, 256) (0, 1)
    list_t2d = list(map(lambda k: list(map(lambda v: test_unit.get_t2d(v), k)), list_ori[:3]))  # (3, 256, 256) (-1, 1)
    list_t2d_norm = list(map(lambda k: list(map(lambda v: norm_01(v), k)), list_t2d))  # (3, 256, 256) (0, 1)
    list_seg = list(map(lambda k: list(map(lambda v: test_unit.get_seg(v), k)), list_t2d_norm))

    #list_diff = [x[1] - x[0] for x in list(zip(list_ori[2], list_ori[1]))]
    list_diff = [torch.abs(torch.div(x[1] - x[0], x[0])) for x in list(zip(list_ori[1], list_ori[2]))]
    for i in range(len(list_diff)):
        diff = list_diff[i]
        diff[diff >= 1] = 1
        list_diff[i] = diff

    to_show = [list_ori[0],
               list(map(lambda x, y: overlap_red(x, y), list_ori_norm[0], list_seg[0])),
               list_ori[1],
               list(map(lambda x, y: overlap_red(x, y), list_ori_norm[1], list_seg[1])),
               list_ori[2],
               list(map(lambda x, y: overlap_red(x, y), list_ori_norm[2], list_seg[2])),
               list_diff,
               list(map(lambda x, y: overlap_red(x, y), list_diff, list_seg[2]))
               ]

    if opt.att:
        to_show = to_show + [list_ori[3]]

    to_show = [torch.cat(x, 2) for x in to_show]
    to_show = [x - x.min() for x in to_show]

    imagesc(np.concatenate([x / x.max() for x in to_show], 1), show=False,
            save=os.path.join("result", opt.prj, str(epoch) + '.jpg'))



# USAGE
# CUDA_VISIBLE_DEVICES=3 python test.py --dataset pain --nepochs 0 601 20 --prj cycle_eff_check --direction a_b --cycle
# CUDA_VISIBLE_DEVICES=1 python test.py --dataset pain --nepochs 0 601 20 --prj wseg1000 --direction a_b
# CUDA_VISIBLE_DEVICES=3 python test.py --dataset TSE_DESS --nepochs 0 601 20 --prj up256patchgan --direction a_b
# CUDA_VISIBLE_DEVICES=0 python test_new.py --dataset TSE_DESS --nepochs 0 601 20 --prj NoResampleResnet9 --direction a_b
