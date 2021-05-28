from __future__ import print_function
import argparse
import os
from utils.data_utils import imagesc
import torch
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
from utils.make_config import load_config
from utils.bone_segmentation.bonesegmentation import BoneSegModel

path = load_config('config.ini', 'Path')


def get_ab(epochs):
    dir_checkpoints = path['checkpoints']
    model_path = dir_checkpoints + "/{}/netG_ab_model_epoch_{}.pth".format(opt.prj, epochs)
    net_ab = torch.load(model_path).to(device)
    return net_ab


def get_ba(epochs):
    dir_checkpoints = path['checkpoints']
    model_path = dir_checkpoints + "/{}/netG_ba_model_epoch_{}.pth".format(opt.prj, epochs)
    net_ba = torch.load(model_path).to(device)
    return net_ba


# Testing settings
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--dataset', default='pain')
parser.add_argument('--prj', type=str, default='cycle0_eff_gloss101010', help='name of the project')
parser.add_argument('--direction', type=str, default='a_b', help='a2b or b2a')
parser.add_argument('--nepochs', nargs='+', default=[0, 600, 20], help='which checkpoints to be interfered with')
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
        root_path = path['dataset']#"dataset/"
        test_set = get_test_set(root_path + opt.dataset, opt.direction, mode='test')

    if not os.path.exists(os.path.join("result", opt.prj)):
        os.makedirs(os.path.join("result", opt.prj))

    seg_model = BoneSegModel()

    device = torch.device("cuda:0")
    for epoch in opt.nepochs:
        a_all = []
        b_all = []
        out_all = []
        for i in [15, 19, 34, 79, 95, 109, 172, 173, 208, 249, 266]:#[1, 19, 25, 29, 34, 39]:#
            # test
            x = test_set.__getitem__(i)
            a = x[0]
            b = x[1]

            a_all.append(a.unsqueeze(0))
            b_all.append(b.unsqueeze(0))
            # model
            input = a.unsqueeze(0).to(device)
            net = get_ab(epoch)
            out = net(input)
            out_all.append(out.detach().cpu())

            #aseg = seg_model.net(img.unsqueeze(0).cuda())[0].detach().cpu().numpy()
            #bseg = seg_model.net(mask.unsqueeze(0).cuda())[0].detach().cpu().numpy()
            #oseg = seg_model.net(out.cuda())[0].detach().cpu().numpy()

        a_all = make_grid(torch.cat(a_all, 0), nrow=12, padding=0)
        b_all = make_grid(torch.cat(b_all, 0), nrow=12, padding=0)
        out_all = make_grid(torch.cat(out_all, 0), nrow=12, padding=0)
        (a_all, b_all, out_all) = (x - x.min() for x in (a_all, b_all, out_all))
        (a_all, b_all, out_all) = (x / x.max() for x in (a_all, b_all, out_all))

        # difference
        cm = plt.get_cmap('viridis')
        dif_all = torch.from_numpy(cm((a_all - out_all))[0, :, :, :3]).type(torch.float32).permute(2, 0, 1)
        imagesc(torch.cat([a_all, b_all, out_all, dif_all], 1), show=False, save=os.path.join("result", opt.prj, str(epoch) + '.jpg'))

