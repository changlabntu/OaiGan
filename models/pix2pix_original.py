import torch
import torch.nn as nn
import torch.optim as optim
from .networks import define_G, define_D
from models.networks import get_scheduler
from models.loss import GANLoss


class Pix2PixModel(nn.Module):
    """
    original Pix2Pix implementation
    """
    def __init__(self, device, opt):
        super(Pix2PixModel, self).__init__()
        self.opt = opt
        self.net_g = define_G(input_nc=opt.input_nc, output_nc=opt.output_nc, ngf=64, netG=opt.netG,
                              norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[])
        self.net_d = define_D(input_nc=opt.input_nc + opt.output_nc, ndf=64, netD=opt.netD)

        self.optimizer_g = optim.Adam(self.net_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_d = optim.Adam(self.net_d.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

        self.net_g_scheduler = get_scheduler(self.optimizer_g, opt)
        self.net_d_scheduler = get_scheduler(self.optimizer_d, opt)

        self.criterionGAN = GANLoss('vanilla').to(device)
        self.criterionL1 = nn.L1Loss().to(device)

    def forward(self, real_a, real_b):
        self.real_a = real_a
        self.real_b = real_b
        self.fake_b = self.net_g(real_a)

    def backward_d(self):
        # train with fake
        fake_ab = torch.cat((self.real_a, self.fake_b), 1)
        pred_fake = self.net_d.forward(fake_ab.detach())
        loss_d_fake = self.criterionGAN(pred_fake, False)

        # train with real
        real_ab = torch.cat((self.real_a, self.real_b), 1)
        pred_real = self.net_d.forward(real_ab)
        loss_d_real = self.criterionGAN(pred_real, True)

        # Combined D loss
        self.loss_d = (loss_d_fake + loss_d_real) * 0.5
        self.loss_d.backward()

    def backward_g(self):
        # First, G(A) should fake the discriminator
        fake_ab = torch.cat((self.real_a, self.fake_b), 1)
        pred_fake = self.net_d(fake_ab)
        loss_g_gan = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        loss_g_l1 = self.criterionL1(self.fake_b, self.real_b) * self.opt.lamb
        self.loss_g = loss_g_gan + loss_g_l1
        self.loss_g.backward()

    def optimize_parameters(self, real_a, real_b):
        self.forward(real_a, real_b)
        ######################
        # (1) Update D network
        ######################
        for param in self.net_d.parameters():
            param.requires_grad = True
        self.optimizer_d.zero_grad()
        self.backward_d()
        self.optimizer_d.step()

        ######################
        # (2) Update G network
        ######################
        for param in self.net_d.parameters():
            param.requires_grad = False
        self.optimizer_g.zero_grad()
        self.backward_g()
        self.optimizer_g.step()
