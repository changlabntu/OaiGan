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
from skimage import io
import torchvision.transforms as transforms
import torch.nn as nn
from engine.base import combine

import numpy as np
import matplotlib.pyplot as plt
cm = plt.get_cmap('viridis')


def to_heatmap(x):
    return cm(x)


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

    c = 3
    x[0, y == c] = 0.1 * x[0, y == c]
    x[2, y == c] = 0.1 * x[2, y == c]

    c = 2
    x[0, y == c] = 0.1 * x[0, y == c]
    x[1, y == c] = 0.1 * x[1, y == c]

    c = 4
    x[0, y == c] = 0.1 * x[0, y == c]
    x[1, y == c] = 0.1 * x[1, y == c]
    return x


class Pix2PixModel:
    def __init__(self, args):
        self.args = args
        self.net_g = None
        self.dir_checkpoints = os.environ.get('LOGS')
        from dataloader.data_multi_old import MultiData as Dataset

        self.test_set = Dataset(root=os.environ.get('DATASET') + args.testset,
                                path=args.direction,
                                opt=args, mode='test')

        #self.seg_model = torch.load(os.environ.get('model_seg')).cuda()
        #self.seg_cartilage = torch.load('submodels/oai_cartilage_384.pth')#model_seg_ZIB.pth')
        self.seg_cartilage = torch.load('submodels/model_seg_ZIB_res18_256.pth')
        self.seg_bone = torch.load('submodels/model_seg_ZIB.pth')
        #self.cartilage = torch.load('submodels/femur_tibia_fc_tc.pth').cuda()
        self.netg_t2d = torch.load('submodels/tse_dess_unet32.pth')

        self.netg_t2d.eval()
        self.seg_cartilage.eval()
        self.seg_bone.eval()

        #self.magic256 = torch.load('/media/ExtHDD01/checkpoints/FlyZ_WpOp/netG_model_epoch_170.pth').cuda()
        #self.magic286 = torch.load('/media/ExtHDD01/checkpoints/FlyZ_WpOp/netG_model_epoch_170.pth').cuda()
        #self.magic286 = torch.load('/media/ExtHDD01/checkpointsold/FlyZ_WpOp286Mask/netG_model_epoch_10.pth').cuda()

        self.device = torch.device("cuda:0")

    def get_model(self,  epoch, eval=False):
        model_path = os.path.join(self.dir_checkpoints, self.args.dataset, self.args.prj, 'checkpoints') + \
               ('/' + self.args.netg + '_model_epoch_{}.pth').format(epoch)
        print(model_path)
        net = torch.load(model_path).to(self.device)
        if eval:
            net.eval()
        else:
            net.train()
        self.net_g = net

    def get_one_output(self, i, xy, alpha=None):
        # inputs
        x = self.test_set.__getitem__(i)
        oriX = x[0].unsqueeze(0).to(self.device)
        oriY = x[1].unsqueeze(0).to(self.device)

        if xy == 'x':
            in_img = oriX
            out_img = oriY
        elif xy == 'y':
            in_img = oriY
            out_img = oriX

        alpha = alpha / 100

        ###
        self.net_g.train()
        for i in range(1):
            try: ## descargan new
                output, output1 = self.net_g(in_img, alpha * torch.ones(1, 2).cuda())
            except:
                try: ## descargan
                    output = self.net_g(in_img, alpha * torch.ones(1, 2).cuda())[0]
                except:
                    try: ## attgan
                        self.net_g.f_size = args.cropsize // 32
                        output = self.net_g(in_img, alpha * torch.ones(1, 1).cuda())[0]
                    except:
                        output = self.net_g(in_img)[0]
        if 0:
            self.net_g.eval()
            try: ## descargan new
                output, output1 = self.net_g(in_img, alpha * torch.ones(1, 2).cuda())
            except:
                try: ## descargan
                    output = self.net_g(in_img, alpha * torch.ones(1, 2).cuda())[0]
                except:
                    try: ## attgan
                        self.net_g.f_size = args.cropsize // 32
                        output = self.net_g(in_img, alpha * torch.ones(1, 1).cuda())[0]
                    except:
                        output = self.net_g(in_img)[0]

        if args.cmb is not None:
            output = combine(1 - output1 + output, in_img, args.cmb)
            #output1 = combine(output1, in_img, args.cmb)
            output = output #+ in_img

        in_img = in_img.detach().cpu()[0, ::]
        out_img = out_img.detach().cpu()[0, ::]
        output = output[0, ::].detach().cpu()

        return in_img, out_img, output

    def get_t2d(self, ori):
        t2d = self.netg_t2d(ori.cuda().unsqueeze(0))[0][0, ::].detach().cpu()
        return t2d

    def get_seg(self, ori):
        bone = self.seg_bone(ori.cuda().unsqueeze(0))
        bone = torch.argmax(bone, 1)[0,::].detach().cpu()

        cartilage = self.seg_cartilage(ori.cuda().unsqueeze(0))
        cartilage = torch.argmax(cartilage, 1)[0,::].detach().cpu()

        #seg[seg == 3] = 0
        #seg[seg == 4] = 0

        seg = 1 * bone

        #cartilage = self.cartilage(norm_01(ori).cuda().unsqueeze(0))
        #cartilage = torch.argmax(cartilage, 1)[0,::].detach().cpu()

        return cartilage

    def get_magic(self, ori):
        magic = self.magic286(ori.cuda().unsqueeze(0))
        magic = magic[0][0, 0, ::].detach().cpu()
        return magic

    def get_all_seg(self, input_ori):
        # normalize
        if self.args.n01:
            input =[]
            for i in range(len(input_ori)):
                temp = []
                for j in range(len(input_ori[0])):
                    temp.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(input_ori[i][j]))
                input.append(temp)
        else:
            input = input_ori

        if self.args.t2d:
            input = list(map(lambda k: list(map(lambda v: self.get_t2d(v), k)), input))  # (3, 256, 256) (-1, 1)
        #list_norm = list(map(lambda k: list(map(lambda v: norm_01(v), k)), input))  # (3, 256, 256) (0, 1)
        list_seg = list(map(lambda k: list(map(lambda v: self.get_seg(v), k)), input))
        #list_seg = list(map(lambda k: list(map(lambda v: self.get_magic(v), k)), input))
        return list_seg


def to_print(to_show, save_name):
    os.makedirs(os.path.join("outputs/results", args.dataset, args.prj), exist_ok=True)
    to_show = [torch.cat(x, len(x[0].shape) - 1) for x in to_show]
    to_show = [x - x.min() for x in to_show]
    to_show = [x / x.max() for x in to_show]
    print(to_show[0].shape)\

    for i in range(len(to_show)):
        if len(to_show[i].shape) == 2:
            to_show[i] = torch.cat([to_show[i].unsqueeze(0)] * 3, 1)

    to_print = np.concatenate([x / x.max() for x in to_show], 1).astype(np.float16)
    imagesc(to_print, show=False, save=save_name)


def seperate_by_seg(x, seg_used, masked, if_absolute, threshold, rgb):
    out = []
    for n in range(len(x)):
        a = 1 * x[n]
        if if_absolute:
            a[a < 0] = 0
        for c in masked:
            a[:, seg_used[n] == c] = 0
        if threshold > 0:
            a[a > threshold] = threshold
        a = a / a.max()
        if rgb:
            a = np.transpose(cm(a[0, ::]), (2, 0, 1))[:3, ::]
            a = torch.from_numpy(a)
        out.append(a)
    return out


# Command Line Argument
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--env', type=str, default=None, help='environment_to_use')
parser.add_argument('--jsn', type=str, default='womac3', help='name of ini file')
parser.add_argument('--dataset', help='name of training dataset')
parser.add_argument('--testset', help='name of testing dataset if different than the training dataset')
parser.add_argument('--prj', type=str, help='NS/unet128_NoNorm')
parser.add_argument('--direction', type=str, help='a_b')
parser.add_argument('--netg', type=str)
parser.add_argument('--resize', type=int)
parser.add_argument('--cropsize', type=int)
parser.add_argument('--bysubject', action='store_true', dest='bysubject', default=False)
parser.add_argument('--t2d', action='store_true', dest='t2d', default=False)
parser.add_argument('--cmb', type=str, default=None, help='way to combine output to the input')
parser.add_argument('--n01', action='store_true', dest='n01')
parser.add_argument('--flip', action='store_true', dest='flip')
parser.add_argument('--eval', action='store_true', dest='eval')
parser.add_argument('--nepochs', default=(30, 40, 10), nargs='+', help='which checkpoints to be interfered with', type=int)
parser.add_argument('--nalpha', default=(0, 100, 1), nargs='+', help='range of additional input parameter for generator', type=int)
parser.add_argument('--mode', type=str, default='dummy')
parser.add_argument('--port', type=str, default='dummy')

with open('outputs/' + parser.parse_args().jsn + '.json', 'rt') as f:
    t_args = argparse.Namespace()
    t_args.__dict__.update(json.load(f))
    args = parser.parse_args(namespace=t_args)

print(args.cropsize)

# environment file
if args.env is not None:
    load_dotenv('.' + args.env)
else:
    load_dotenv('.env')

if len(args.nepochs) == 1:
    args.nepochs = [args.nepochs[0], args.nepochs[0]+1, 1]
if len(args.nalpha) == 1:
    args.nalpha = [args.nalpha[0], args.nalpha[0]+1, 1]

test_unit = Pix2PixModel(args=args)
print(len(test_unit.test_set))

for epoch in range(*args.nepochs):
    #try:
    test_unit.get_model(epoch, eval=args.eval)
    for ii in range(1):#range(len(test_unit.test_set)):
        #args.irange = [ii]

        seg0_all = []
        seg1_all = []

        for alpha in np.linspace(*args.nalpha):
            out_xy = list(map(lambda v: test_unit.get_one_output(v, 'x', alpha), args.irange))
            out_xy = list(zip(*out_xy))

            seg_xy = test_unit.get_all_seg(out_xy)
            diff_xy = [(x[1] - x[0]) for x in list(zip(out_xy[2], out_xy[0]))]

            #diff_xy[0][0, 0, 0] = 1.5
            #diff_xy[0][0, 0, 1] = -1.5

            #for i in range(len(diff_xy)):
                #diff_xy[i][diff_xy[i] > 0.2] = 0.2
                #diff_xy[i][diff_xy[i] < -0.2] = -0.2

                #diff_xy[i] = diff_xy[i] - diff_xy[i].min()
            #    diff_xy[i] = diff_xy[i] / diff_xy[i].max()

            ooo = [x - x.min() for x in out_xy[0]]

            tag = True
            x0 = seperate_by_seg(x=ooo, seg_used=seg_xy[0], masked=[0, 2, 4], if_absolute=tag, threshold=0, rgb=tag)
            diffseg0 = seperate_by_seg(x=diff_xy, seg_used=seg_xy[2], masked=[0, 2, 4], if_absolute=tag, threshold=0, rgb=tag)
            diffseg1 = seperate_by_seg(x=diff_xy, seg_used=seg_xy[2], masked=[1, 3], if_absolute=tag, threshold=0, rgb=tag)

            seg0_all.append(np.expand_dims(np.concatenate([x[:1, ::] for x in diffseg0], 0), 3))
            seg1_all.append(np.expand_dims(np.concatenate([x[:1, ::] for x in diffseg1], 0), 3))

            #diffseg0 = [abs(x) for x in diffseg0]
            #diffseg1 = [abs(x) for x in diffseg1]

            to_show = [out_xy[0],
                       #seg_xy[0],
                       #list(map(lambda x, y: overlap_red(x, y), out_xy[0], seg_xy[0])),
                       #out_xy[1],
                       #seg_xy[1],
                       #list(map(lambda x, y: overlap_red(x, y), out_xy[1], seg_xy[1])),
                       out_xy[2],
                       #seg_xy[2],
                       #diff_xy,
                       #x_seg,
                       #x0,
                       diffseg0,
                       diffseg1
                       #list(map(lambda x, y: overlap_red(x, y), diff_xy, seg_xy[2])),
                       ]

            to_print(to_show, save_name=os.path.join("outputs/results", args.dataset, args.prj,
                                                     str(epoch) + '_' + str(alpha) + '_' + str(ii).zfill(4) + '.jpg'))

        seg0_all = np.concatenate(seg0_all, 3)
        seg1_all = np.concatenate(seg1_all, 3)

        print(seg0_all.shape)
        print(seg1_all.shape)

        #np.save('seg0.npy', seg0_all)
        #np.save('seg1.npy', seg1_all)

    #except:
    #    print('failed for some reason')


# USAGE
# CUDA_VISIBLE_DEVICES=0 python test.py --jsn default --dataset pain --nalpha 0 100 2  --prj VryAtt
# CUDA_VISIBLE_DEVICES=0 python test.py --jsn default --dataset TSE_DESS --nalpha 0 100 1  --prj VryCycleUP --net netGXY
# python testoai.py --jsn womac3 --direction a_b --prj N01/DescarMul/ --cropsize 384 --n01 --cmb mul


