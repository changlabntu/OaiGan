# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File  : networks.py
# @Author: Jeffrey, Jehovah
# @Date  : 19-9

import torch
import torch.nn as nn
#from config.SAND_pix_opt import TrainOptions
from models.genre.base_network import BaseNetwork
from models.genre.blocks.unet_block import UNetDown, SPADEUp


class SPADEUNet(BaseNetwork):
    def __init__(self, opt, in_channels=3, out_channels=3):
        super(SPADEUNet, self).__init__()
        self.opt = opt
        self.down1 = nn.Conv2d(in_channels, 64, 4, 2, 1)
        self.down2 = UNetDown(64, 128, norm_fun=nn.InstanceNorm2d)
        self.down3 = UNetDown(128, 256, norm_fun=nn.InstanceNorm2d)
        self.down4 = UNetDown(256, 512, norm_fun=nn.InstanceNorm2d)
        self.down5 = UNetDown(512, 512, norm_fun=nn.InstanceNorm2d)
        self.down6 = UNetDown(512, 512, norm_fun=nn.InstanceNorm2d)
        self.down7 = UNetDown(512, 512, norm_fun=nn.InstanceNorm2d)
        if self.opt.input_size > 256:
            self.down8 = UNetDown(512, 512, norm_fun=nn.InstanceNorm2d)
            self.down9 = UNetDown(512, 512, normalize=False)
        else:
            self.down8 = UNetDown(512, 512, normalize=False)

        self.up0 = SPADEUp(self.opt, 512, 512, 0.5)
        self.up1 = SPADEUp(self.opt, 1024, 512, 0.5)
        self.up2 = SPADEUp(self.opt, 1024, 512, 0.5)
        self.up3 = SPADEUp(self.opt, 1024, 512)
        if self.opt.input_size > 256:
            self.up3_plus = SPADEUp(self.opt, 1024, 512)
        self.up4 = SPADEUp(self.opt, 1024, 256)
        self.up5 = SPADEUp(self.opt, 512, 128)
        self.up6 = SPADEUp(self.opt, 256, 64)

        self.final = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x, parsing):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        if self.opt.input_size <= 256:
            u0 = self.up0(d8, parsing)
            u1 = self.up1(u0, parsing, d7)
            u2 = self.up2(u1, parsing, d6)
            u3 = self.up3(u2, parsing, d5)
            u4 = self.up4(u3, parsing, d4)
            u5 = self.up5(u4, parsing, d3)
            u6 = self.up6(u5, parsing, d2)
        else:
            d9 = self.down9(d8)
            u0 = self.up0(d9, parsing)
            u1 = self.up1(u0, parsing, d8)
            u2 = self.up2(u1, parsing, d7)
            u3 = self.up3(u2, parsing, d6)
            u3_p = self.up3_plus(u3, parsing, d5)
            u4 = self.up4(u3_p, parsing, d4)
            u5 = self.up5(u4, parsing, d3)
            u6 = self.up6(u5, parsing, d2)

        u7 = torch.cat([u6, d1], dim=1)
        u8 = self.final(u7)

        return u8


class SPADEUNet_YPar(BaseNetwork):
    def __init__(self, opt, img_channel, par_channel, out_channels=3):
        super(SPADEUNet_YPar, self).__init__()
        self.opt = opt
        self.down_rgb = nn.Conv2d(img_channel, 64, 4, 2, 1)
        self.down_par = nn.Conv2d(par_channel, 64, 4, 2, 1)
        self.down2 = UNetDown(128, 128, norm_fun=nn.InstanceNorm2d)
        self.down3 = UNetDown(128, 256, norm_fun=nn.InstanceNorm2d)
        self.down4 = UNetDown(256, 512, norm_fun=nn.InstanceNorm2d)
        self.down5 = UNetDown(512, 512, norm_fun=nn.InstanceNorm2d)
        self.down6 = UNetDown(512, 512, norm_fun=nn.InstanceNorm2d)
        self.down7 = UNetDown(512, 512, norm_fun=nn.InstanceNorm2d)
        self.down8 = UNetDown(512, 512, normalize=False)
        self.up0 = SPADEUp(self.opt, 512, 512, 0.5, first=True)
        self.up1 = SPADEUp(self.opt, 1024, 512, 0.5)
        self.up2 = SPADEUp(self.opt, 1024, 512, 0.5)
        self.up3 = SPADEUp(self.opt, 1024, 512)
        self.up4 = SPADEUp(self.opt, 1024, 256)
        self.up5 = SPADEUp(self.opt, 512, 128)
        self.up6 = SPADEUp(self.opt, 256, 64)

        self.final_rgb = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),
            nn.Tanh()
        )
        self.final_par = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, par_channel, 4, 2, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x, x_parsing, y_parsing):
        d1_rgb = self.down_rgb(x)
        d1_par = self.down_par(x_parsing)
        d1 = torch.cat([d1_rgb, d1_par], dim=1)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u0 = self.up0(d8, y_parsing)
        u1 = self.up1(u0, y_parsing, d7)
        u2 = self.up2(u1, y_parsing, d6)
        u3 = self.up3(u2, y_parsing, d5)
        u4 = self.up4(u3, y_parsing, d4)
        u5 = self.up5(u4, y_parsing, d3)
        u6, gamma_beta = self.up6(u5, y_parsing, d2, gamma_mode="final")

        u7_rgb = torch.cat([u6, d1_rgb], dim=1)
        u7_par = torch.cat([u6, d1_par], dim=1)
        u8_rgb = self.final_rgb(u7_rgb)
        u8_par = self.final_par(u7_par)

        return u8_rgb, u8_par
        # return u8_rgb


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()  # add_help=False)
    # Env
    parser.add_argument('--input_size', type=float, default=256)
    parser.add_argument('--parsing_nc', type=int, default=19,
                        help='# of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.')
    parser.add_argument('--norm_G', type=str, default='spectralspadebatch3x3',
                        help='instance normalization or batch normalization')
    parser.add_argument('--spade_mode', type=str, default="res2", choices=('org', 'res', 'concat', 'res2'),
                        help='type of spade shortcut connection : |org|res|concat|res2|')
    parser.add_argument('--use_en_feature', action='store_true', help='cat encoder feature to parsing')
    parser.add_argument('--mode', type=str, default='dummy')
    parser.add_argument('--port', type=str, default='dummy')
    opt = parser.parse_args()

    #opt = TrainOptions().parse()
    #opt.input_size = 256
    #opt.spade_mode = 'res'
    #opt.norm_G = 'spectraloutterbatch3x3'
    # opt = SPADEGenerator.modify_commandline_options(opt)
    style = torch.randn([2, 1, 256, 256])
    x = torch.randn([2, 19, 256, 256])

    m = SPADEUNet(opt=opt, in_channels=1, out_channels=1)
    m(style, x)

    # model = OutterUNet(opt, in_channels=3, out_channels=1)
    #y_identity, gamma_beta = model.forward(style)
    #hat_y, _ = model.forward(x, use_basic=False, gamma=gamma_beta[0], beta=gamma_beta[1])
    #print(hat_y.size())
