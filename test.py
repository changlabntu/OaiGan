from __future__ import print_function
import argparse
import os
from utils.data_utils import imagesc
import torch
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt


def get_model(epochs):
    dir_checkpoints = '/media/ghc/GHc_data1/checkpoints'
    model_path = dir_checkpoints + "/{}/netG_model_epoch_{}.pth".format(opt.prj, epochs)
    net_g = torch.load(model_path).to(device)
    return net_g


def get_D(epochs):
    dir_checkpoints = '/media/ghc/GHc_data1/checkpoints'
    model_path = dir_checkpoints + "/{}/netD_model_epoch_{}.pth".format(opt.prj, epochs)
    net_d = torch.load(model_path).to(device)
    return net_d


# Testing settings
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--dataset', default='pain', help='facades')
parser.add_argument('--prj', type=str, default='test', help='name of the project')
parser.add_argument('--direction', type=str, default='a_b', help='a2b or b2a')
parser.add_argument('--nepochs', nargs='+', default=[20, 200, 20], help='which checkpoints to be interfered with')
parser.add_argument('--mode', type=str, default='dummy')
parser.add_argument('--port', type=str, default='dummy')
opt = parser.parse_args()
opt.prj = opt.dataset + '_' + opt.prj
opt.nepochs = range(int(opt.nepochs[0]), int(opt.nepochs[1]), int(opt.nepochs[2]))
print(opt)

if __name__ == '__main__':
    if opt.dataset == 'dess':
        from dataloader.data_dess_segmentation import DatasetDessSegmentation
        test_set = DatasetDessSegmentation(mode='val', direction='a2b')
    else:
        from dataloader.data import get_test_set
        root_path = "dataset/"
        test_set = get_test_set(root_path + opt.dataset, opt.direction, mode='test')

    if not os.path.exists(os.path.join("result", opt.prj)):
        os.makedirs(os.path.join("result", opt.prj))

    device = torch.device("cuda:0")
    for epochs in opt.nepochs:
        a_all = []
        b_all = []
        out_all = []
        for i in [1612, 323, 1279, 1134, 1884, 494, 1371, 163, 451, 1729, 1062, 1496]:
            # test
            x = test_set.__getitem__(i)
            img = x[0]
            mask = x[1]
            a_all.append(img.unsqueeze(0))
            b_all.append(mask.unsqueeze(0))
            # model
            input = img.unsqueeze(0).to(device)
            net_g = get_model(epochs)
            out = net_g(input)
            out_all.append(out.detach().cpu())

        a_all = make_grid(torch.cat(a_all, 0), nrow=12, padding=0)
        b_all = make_grid(torch.cat(b_all, 0), nrow=12, padding=0)
        out_all = make_grid(torch.cat(out_all, 0), nrow=12, padding=0)
        (a_all, b_all, out_all) = (x - x.min() for x in (a_all, b_all, out_all))
        (a_all, b_all, out_all) = (x / x.max() for x in (a_all, b_all, out_all))

        # difference
        cm = plt.get_cmap('viridis')
        dif_all = torch.from_numpy(cm((a_all - out_all))[0, :, :, :3]).type(torch.float32).permute(2, 0, 1)
        imagesc(torch.cat([a_all, b_all, out_all, dif_all], 1), show=False, save=os.path.join("result", opt.prj, str(epochs) + '.jpg'))

