import torch
import torch.nn as nn
import torch.optim as optim
from .networks import define_G, define_D
from models.networks import get_scheduler
from models.loss import GANLoss
from math import log10
import time, os
import pytorch_lightning as pl
from models.cyclegan.models import GeneratorResNet, Discriminator


def _weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


class Pix2PixModel(pl.LightningModule):
    """
    There is a lot of patterned noise and other failures when using lightning
    """
    def __init__(self, hparams, train_loader, test_loader, checkpoints):
        super(Pix2PixModel, self).__init__()
        print('using pix2pix.py')
        # initialize
        self.tini = time.time()
        self.epoch = 0
        self.avg_psnr = 0

        # opts
        self.hparams = hparams
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.net_g = define_G(input_nc=hparams.input_nc, output_nc=hparams.output_nc, ngf=64, netG=hparams.netG,
                              norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[])
        #self.net_d = define_D(input_nc=hparams.input_nc + hparams.output_nc, ndf=64, netD=hparams.netD)

        #self.net_d = Discriminator(input_shape=(6, 256, 256))  ##  use the discriminator from CycleGAN PatchGAN
        if hparams.netD == 'patchgan':
            self.net_d = Discriminator(input_shape=(6, 256, 256))
        else:
            self.net_d = define_D(input_nc=hparams.output_nc * 2, ndf=64, netD=hparams.netD)


        self.net_g = self.net_g.apply(_weights_init)
        self.net_d = self.net_d.apply(_weights_init)

        [self.optimizer_d, self.optimizer_g], [] = self.configure_optimizers()
        self.net_g_scheduler = get_scheduler(self.optimizer_g, hparams)
        self.net_d_scheduler = get_scheduler(self.optimizer_d, hparams)

        self.dir_checkpoints = checkpoints

        self.criterionL1 = nn.L1Loss().cuda()
        if hparams.gan_mode == 'vanilla':
            self.criterionGAN = nn.BCEWithLogitsLoss()
        else:
            self.criterionGAN = GANLoss(hparams.gan_mode).cuda()

    def configure_optimizers(self):
        self.optimizer_g = optim.Adam(self.net_g.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, 0.999))
        self.optimizer_d = optim.Adam(self.net_d.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, 0.999))

        return [self.optimizer_d, self.optimizer_g], []

    def backward_g(self, real_images, conditioned_images):
        self.net_g.zero_grad()
        fake_images = self.net_g(conditioned_images)
        #print(torch.cat((fake_images, conditioned_images), 1).shape)
        disc_logits = self.net_d(torch.cat((fake_images, conditioned_images), 1))
        adversarial_loss = self.criterionGAN(disc_logits, torch.ones_like(disc_logits))

        # calculate reconstruction loss
        recon_loss = self.criterionL1(fake_images, real_images)
        lambda_recon = self.hparams.lamb

        loss_g = adversarial_loss + lambda_recon * recon_loss
        return loss_g

    def backward_d(self, real_images, conditioned_images):
        self.net_d.zero_grad()
        fake_images = self.net_g(conditioned_images).detach()

        fake_logits = self.net_d(torch.cat((fake_images, conditioned_images), 1))
        real_logits = self.net_d(torch.cat((real_images, conditioned_images), 1))

        fake_loss = self.criterionGAN(fake_logits, torch.zeros_like(fake_logits))
        real_loss = self.criterionGAN(real_logits, torch.ones_like(real_logits))

        # Combined D loss
        loss_d = (real_loss + fake_loss) * 0.5
        return loss_d

    def training_step(self, batch, batch_idx, optimizer_idx):
        condition, real = batch
        if optimizer_idx == 0:
            #self.net_d.zero_grad()
            #for param in self.net_d.parameters():
            #    param.requires_grad = True
            loss_d = self.backward_d(real, condition)
            self.log('loss_d', loss_d, on_step=False, on_epoch=True,
                     prog_bar=True, logger=True, sync_dist=True)
            return loss_d

        if optimizer_idx == 1:
            #self.net_g.zero_grad()
            #for param in self.net_d.parameters():
            #    param.requires_grad = False
            loss_g = self.backward_g(real, condition)
            self.log('loss_g', loss_g, on_step=False, on_epoch=True,
                     prog_bar=True, logger=True, sync_dist=True)
            return loss_g

    def training_epoch_end(self, outputs):
        hparams = self.hparams
        self.net_g_scheduler.step()
        self.net_d_scheduler.step()
        # checkpoint
        dir_checkpoints = self.dir_checkpoints
        if self.epoch % 20 == 0:
            if not os.path.exists(dir_checkpoints):
                os.mkdir(dir_checkpoints)
            if not os.path.exists(os.path.join(dir_checkpoints, hparams.prj)):
                os.mkdir(os.path.join(dir_checkpoints, hparams.prj))
            net_g_model_out_path = dir_checkpoints + "/{}/netG_model_epoch_{}.pth".format(hparams.prj, self.epoch)
            net_d_model_out_path = dir_checkpoints + "/{}/netD_model_epoch_{}.pth".format(hparams.prj, self.epoch)
            torch.save(self.net_g, net_g_model_out_path)
            torch.save(self.net_d, net_d_model_out_path)
            print("Checkpoint saved to {}".format(dir_checkpoints + '/' + hparams.prj))

        self.epoch += 1
        self.tini = time.time()
        self.avg_psnr = 0
