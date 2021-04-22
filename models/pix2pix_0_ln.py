import torch
import torch.nn as nn
import torch.optim as optim
from .networks import define_G, define_D
from models.networks import get_scheduler
from models.loss import GANLoss
from math import log10
import time, os
import pytorch_lightning as pl


class Pix2PixModel(pl.LightningModule):
    """
    I refactor it to lightning
    """
    def __init__(self, hparams, train_loader, test_loader):
        super(Pix2PixModel, self).__init__()
        print('using pix2pix_0')
        # initialize
        self.tini = time.time()
        self.epoch = 0
        self.avg_psnr = 0

        # opts
        self.hparams = hparams
        self.opt = hparams
        opt = hparams
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.net_g = define_G(input_nc=opt.input_nc, output_nc=opt.output_nc, ngf=64, netG=opt.netG,
                              norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[])
        self.net_d = define_D(input_nc=opt.input_nc + opt.output_nc, ndf=64, netD=opt.netD)

        [self.optimizer_g, self.optimizer_d], [] = self.configure_optimizers()
        self.net_g_scheduler = get_scheduler(self.optimizer_g, opt)
        self.net_d_scheduler = get_scheduler(self.optimizer_d, opt)

        self.criterionGAN = GANLoss('vanilla').cuda()
        self.criterionL1 = nn.L1Loss().cuda()

        #self.device = device

    def configure_optimizers(self):
        self.optimizer_g = optim.Adam(self.net_g.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        self.optimizer_d = optim.Adam(self.net_d.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))

        return [self.optimizer_g, self.optimizer_d], []

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
        loss_d = (loss_d_fake + loss_d_real) * 0.5
        return loss_d

    def backward_g(self):
        # First, G(A) should fake the discriminator
        fake_ab = torch.cat((self.real_a, self.fake_b), 1)
        pred_fake = self.net_d(fake_ab)
        loss_g_gan = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        loss_g_l1 = self.criterionL1(self.fake_b, self.real_b) * self.opt.lamb
        loss_g = loss_g_gan + loss_g_l1
        return loss_g

    def training_step(self, batch, batch_idx, optimizer_idx):
        if optimizer_idx == 0:
            self.real_a, self.real_b = batch[0].cuda(), batch[1].cuda()
            self.fake_b = self.net_g(self.real_a)
            # update G
            for param in self.net_d.parameters():
                param.requires_grad = False
            loss_g = self.backward_g()
            self.log('loss_g', loss_g, on_step=False, on_epoch=True,
                     prog_bar=True, logger=True, sync_dist=True)
            return loss_g

        if optimizer_idx == 1:
            # update D
            for param in self.net_d.parameters():
                param.requires_grad = True
            loss_d = self.backward_d()
            self.log('loss_d', loss_d, on_step=False, on_epoch=True,
                     prog_bar=True, logger=True, sync_dist=True)
            return loss_d

    def training_epoch_end(self, outputs):
        self.net_g_scheduler.step()
        self.net_d_scheduler.step()
        # update_learning_rate(net.module.net_g_scheduler, net.module.optimizer_g)
        # update_learning_rate(net.module.net_d_scheduler, net.module.optimizer_d)

    def validation_step(self, batch, batch_idx=0):
        input, target = batch[0].cuda(), batch[1].cuda()
        prediction = self.net_g(input)
        mse = nn.MSELoss().cuda()(prediction, target)
        psnr = 10 * log10(1 / mse.item())
        self.avg_psnr += psnr
        return 0

    def validation_epoch_end(self, outputs):
        opt = self.opt
        if opt.legacy:
            print("===> Epoch[{}] T: {:.2f} PSNR: {:.4f} Loss_D: {:.4f} Loss_G: {:.4f}".format(
                  self.epoch, time.time() - self.tini, self.avg_psnr / len(self.test_loader),
                  self.loss_d_epoch, self.loss_g_epoch))

        # checkpoint
        dir_checkpoints = '/media/ghc/GHc_data1/checkpoints'
        if self.epoch % 20 == 0:
            if not os.path.exists(dir_checkpoints):
                os.mkdir(dir_checkpoints)
            if not os.path.exists(os.path.join(dir_checkpoints, opt.prj)):
                os.mkdir(os.path.join(dir_checkpoints, opt.prj))
            net_g_model_out_path = dir_checkpoints + "/{}/netG_model_epoch_{}.pth".format(opt.prj, self.epoch)
            net_d_model_out_path = dir_checkpoints + "/{}/netD_model_epoch_{}.pth".format(opt.prj, self.epoch)
            torch.save(self.net_g, net_g_model_out_path)
            torch.save(self.net_d, net_d_model_out_path)
            print("Checkpoint saved to {}".format(dir_checkpoints + '/' + opt.prj))

        self.epoch += 1
        self.tini = time.time()
        self.avg_psnr = 0

    # vanilla pytorch
    def training_loop(self, training_data_loader):
        loss_d_epoch = 0
        loss_g_epoch = 0

        # training epoch
        for iteration, batch in enumerate(training_data_loader, 1):
            # loss G
            loss_g = self.training_step(batch=batch, batch_idx=0, optimizer_idx=0)
            loss_g.backward()
            self.optimizer_g.step()
            self.optimizer_g.zero_grad()
            loss_d = self.training_step(batch=batch, batch_idx=0, optimizer_idx=1)
            # loss D
            loss_d.backward()
            self.optimizer_d.step()
            self.optimizer_d.zero_grad()

            # stastistics
            loss_d_epoch += loss_d
            loss_g_epoch += loss_g

        return loss_d_epoch.item() / (iteration + 1), loss_g_epoch.item() / (iteration + 1)

    def validation_loop(self):
        for batch in self.test_loader:
            self.validation_step(batch)

    def overall_loop(self):
        opt = self.opt
        for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
            # train
            self.loss_d_epoch, self.loss_g_epoch = self.training_loop(self.train_loader)
            self.training_epoch_end(outputs=None)

            # test
            self.validation_loop()
            self.validation_epoch_end(outputs=None)
