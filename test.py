
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
from utils.data_utils import norm_01

load_dotenv('.env')


def get_model(epochs, name, dir_checkpoints, device):
    model_path = dir_checkpoints + "{}/{}_model_epoch_{}.pth".format(opt.prj, name, epochs)
    net = torch.load(model_path).to(device)
    return net


def overlap_red(x0, y0):
    y = 1 * y0
    x = 1 * x0.abs()

    c = 0
    x[1, y == c] = 0
    x[2, y == c] = 0

    c = 2
    x[0, y == c] = 0
    x[1, y == c] = 0

    c = 1
    x[0, y == c] = 0
    x[2, y == c] = 0
    return x


class Pix2PixModel:
    def __init__(self, opt):
        self.opt = opt
        self.dir_checkpoints = os.environ.get('CHECKPOINTS')
        from dataloader.data import DatasetFromFolder as Dataset

        if opt.testset:
            self.test_set = Dataset(os.environ.get('DATASET') + opt.testset + '/test/', opt, mode='test')
        else:
            self.test_set = Dataset(os.environ.get('DATASET') + opt.dataset + '/test/', opt, mode='test')

        if not os.path.exists(os.path.join("result", opt.prj)):
            os.makedirs(os.path.join("result", opt.prj))

        self.seg_model = torch.load(os.environ.get('model_seg')).cuda()
        self.netg_t2d = torch.load(os.environ.get('model_t2d')).cuda()

        self.device = torch.device("cuda:0")

        #self.irange = [15, 19, 34, 79, 95, 109, 172, 173, 208, 249, 266]  # pain
        # irange = [102, 109, 128, 190, 193, 235, 252, 276, 318, 335]

    def get_one_output(self, i, xy, alpha=None):
        opt = self.opt
        x = self.test_set.__getitem__(i)

        # model
        oriX = x[0].unsqueeze(0).to(self.device)
        oriY = x[1].unsqueeze(0).to(self.device)

        if not opt.cycle:
            net_g = get_model(epoch, 'netG', self.dir_checkpoints, self.device)
        else:
            net_g = get_model(epoch, 'netG_ab', self.dir_checkpoints, self.device)

        if xy == 'x':
            in_img = oriX
            out_img = oriY
        elif xy == 'y':
            in_img = oriY
            out_img = oriX

        alpha = alpha

        try:
            output = net_g(in_img, alpha * torch.ones(1, 2).cuda())#, res=True)
        except:
            try:
                output = net_g(in_img, alpha * torch.ones(1, 1).cuda())
            except:
                output = net_g(in_img)

        in_img = in_img.detach().cpu()[0, ::]
        out_img = out_img.detach().cpu()[0, ::]
        output = output[0][0, ::].detach().cpu()

        if self.opt.att:
            return in_img, out_img, output, output[1][0, ::].detach().cpu()
        else:
            return in_img, out_img, output

    def get_t2d(self, ori):
        t2d = self.netg_t2d(ori.cuda().unsqueeze(0))[0][0, ::].detach().cpu()
        return t2d

    def get_seg(self, ori):
        seg = self.seg_model(ori.cuda().unsqueeze(0))[0]
        seg = torch.argmax(seg, 1)[0,::].detach().cpu()
        return seg

    def get_all_seg(self, input):
        list_t2d = list(map(lambda k: list(map(lambda v: self.get_t2d(v), k)), input))  # (3, 256, 256) (-1, 1)
        list_t2d_norm = list(map(lambda k: list(map(lambda v: norm_01(v), k)), list_t2d))  # (3, 256, 256) (0, 1)
        list_seg = list(map(lambda k: list(map(lambda v:self.get_seg(v), k)), list_t2d_norm))
        return list_seg

# Testing settings
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--dataset', default='pain', help='name of training dataset')
parser.add_argument('--testset', default='pain', help='name of testing dataset if different than the training dataset')
parser.add_argument('--prj', type=str, default='NS2AttG', help='name of the project')
parser.add_argument('--direction', type=str, default='a_b', help='a2b or b2a')
parser.add_argument('--resize', type=int, default=286)
parser.add_argument('--flip', action='store_true', dest='flip', default=False)
parser.add_argument('--nepochs', nargs='+', default=[100, 110, 10], help='which checkpoints to be interfered with')
parser.add_argument('--nalpha', nargs='+', default=[0, 1, 1], help='list of alphas')
parser.add_argument('--mode', type=str, default='dummy')
parser.add_argument('--port', type=str, default='dummy')
parser.add_argument('--att', action='store_true', dest='att', default=False)
parser.add_argument('-board-cycle', action='store_true', dest='cycle', default=False)
parser.add_argument('--t2d', action='store_true', dest='t2d', default=True)

opt = parser.parse_args()
opt.prj = opt.dataset + '_' + opt.prj
opt.nepochs = range(int(opt.nepochs[0]), int(opt.nepochs[1]), int(opt.nepochs[2]))
print(opt.nalpha)



test_unit = Pix2PixModel(opt=opt)
#irange = list(np.random.randint(0, 250, 10))#
#irange = [15, 19, 34, 79, 95, 109, 172, 173, 208, 249, 266]
irange = [15, 19, 34, 79, 95, 109, 172, 173, 208, 249, 266][5:]  # pain

zz = list(opt.nalpha)

for epoch in opt.nepochs:
    for alpha in np.linspace(float(zz[0]), float(zz[1]), int(zz[2])):
        out_xy = list(map(lambda v: test_unit.get_one_output(v, 'x', alpha), irange))
        out_yx = list(map(lambda v: test_unit.get_one_output(v, 'y', alpha), irange))

        out_xy = list(zip(*out_xy))
        out_yx = list(zip(*out_yx))

        #out_xy = list(map(lambda k: list(map(lambda v: norm_01(v), k)), out_xy[:3]))
        #out_yx = list(map(lambda k: list(map(lambda v: norm_01(v), k)), out_yx[:3]))

        seg_xy = test_unit.get_all_seg(out_xy)
        seg_yx = test_unit.get_all_seg(out_yx)


        diff_xy = [(x[1] - x[0]) for x in list(zip(out_xy[2], out_xy[0]))]
        diff_yx = [(x[1] - x[0]) for x in list(zip(out_yx[2], out_yx[0]))]

        diff_xy[0][0, 0, 0] = 2
        diff_xy[0][0, 0, 1] = -2
        diff_yx[0][0, 0, 0] = 2
        diff_yx[0][0, 0, 1] = -2

        #out_xy[2] = [(x<-0.2)/1 for x in out_xy[2]]
        #out_yx[2] = [(x<-0.2)/1 for x in out_yx[2]]
        #diff_xy = [(x>0.5)/1 for x in diff_xy]
        #diff_yx = [(x>0.5)/1 for x in diff_yx]

        to_show = [out_xy[0],
                   out_xy[2],
                   diff_xy,
                   list(map(lambda x, y: overlap_red(x, y), diff_xy, seg_xy[2])),
                   out_yx[0],
                   out_yx[2],
                   diff_yx,
                   list(map(lambda x, y: overlap_red(x, y), diff_yx, seg_yx[2])),
                   #list(map(lambda x, y: overlap_red(x, y), list_ori_norm[0], list_seg[0])),
                   #list_ori[1],
                   #list(map(lambda x, y: overlap_red(x, y), list_ori_norm[1], list_seg[1])),
                   #list_ori[2],
                   #list(map(lambda x, y: overlap_red(x, y), list_ori_norm[2], list_seg[2])),
                   #list_diff
                   #list(map(lambda x, y: overlap_red(x, y), list_diff, list_seg[2]))
                   ]

        to_show = [torch.cat(x, 2) for x in to_show]
        to_show = [x - x.min() for x in to_show]

        imagesc(np.concatenate([x / x.max() for x in to_show], 1), show=False,
                save=os.path.join("result", opt.prj, str(epoch) + '_' + str(alpha) +'.jpg'))



# USAGE
# CUDA_VISIBLE_DEVICES=3 python test.py --dataset pain --nepochs 0 601 20 --prj cycle_eff_check --direction a_b --cycle
# CUDA_VISIBLE_DEVICES=1 python test.py --dataset pain --nepochs 0 601 20 --prj wseg1000 --direction a_b
# CUDA_VISIBLE_DEVICES=3 python test.py --dataset TSE_DESS --nepochs 0 601 20 --prj up256patchgan --direction a_b

# CUDA_VISIBLE_DEVICES=0 python test.py --dataset pain --nepochs 0 601 20 --prj bysubjectwgan --direction a_b --resize 286

# CUDA_VISIBLE_DEVICES=0 python test.py --dataset TSE_DESS --nepochs 0 601 20 --prj NoResampleResnet9 --direction a_b --ressize 0


# CUDA_VISIBLE_DEVICES=0 python test.py --dataset painfull --nepochs 0 601 10 --prj b6DattganDG01--direction a_b --resize 286

# CUDA_VISIBLE_DEVICES=0 python test.py --dataset pain --nepochs 0 910 10 --nalpha 0 1 2  --prj Try3descarG --direction a_b --resize 286

