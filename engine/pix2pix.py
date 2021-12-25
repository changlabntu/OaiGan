import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from models.networks import define_G, define_D
from models.networks import get_scheduler
from models.loss import GANLoss
from math import log10
import time, os
import pytorch_lightning as pl
from utils.metrics_segmentation import SegmentationCrossEntropyLoss
from utils.metrics_classification import CrossEntropyLoss, GetAUC
from utils.data_utils import *


def mix_lr(oriX, oriY):
    dx = oriX.shape[2] // 2

    oriA = 1 * oriX
    oriB = 1 * oriY
    oriA[:, :, :, :dx] = oriX[:, :, :, :dx]
    oriA[:, :, :, dx:] = oriY[:, :, :, dx:]
    oriB[:, :, :, :dx] = oriY[:, :, :, :dx]
    oriB[:, :, :, dx:] = oriX[:, :, :, dx:]
    oriX = oriA
    oriY = oriB
    return oriX, oriY


def _weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)
        torch.nn.init.constant_(m.bias, 0)


class MRPretrained(nn.Module):
    def __init__(self):
        super(MRPretrained, self).__init__()
        self.features = getattr(torchvision.models, 'vgg11')(pretrained=True).features
        self.fmap_c = 512
        # fusion part
        self.classifier_cat = nn.Conv2d(self.fmap_c * 23, 2, 1, 1, 0)
        self.classifier_max = nn.Conv2d(self.fmap_c, 2, 1, 1, 0)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):   # (23, 3, 256, 256)
        x0 = x[0]
        x1 = x[1]
        # dummies
        out = None  # output of the model
        features = None  # features we want to further analysis
        B = 1
        # features
        x0 = self.features(x0)  # (23, 512, 7, 7)
        x1 = self.features(x1)  # (23, 512, 7, 7)
        # fusion
        x0 = self.avg(x0)  # (B*23, 512, 1, 1)
        x0 = x0.view(B, x0.shape[0] // B, x0.shape[1], x0.shape[2], x0.shape[3])  # (B, 23, 512, 1, 1)
        x1 = self.avg(x1)  # (B*23, 512, 1, 1)
        x1 = x1.view(B, x1.shape[0] // B, x1.shape[1], x1.shape[2], x1.shape[3])  # (B, 23, 512, 1, 1)

        x0, _ = torch.max(x0, 1)
        x1, _ = torch.max(x1, 1)
        out = self.classifier_max(x0 - x1)  # (Classes)

        out = out[:, :, 0, 0]
        return out, features


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

        # hyperparameters
        hparams = {x:vars(hparams)[x] for x in vars(hparams).keys() if x not in hparams.not_tracking_hparams}
        hparams.pop('not_tracking_hparams', None)
        self.hparams = hparams
        self.hparams.update(hparams)
        print(self.hparams)
        self.save_hyperparameters(self.hparams)
        self.best_auc = 0

        # input options
        self.GZ = True

        # GENERATOR
        if self.hparams.netG == 'attgan':
            from models.AttGAN.attgan import Generator
            print('use attgan discriminator')
            self.net_g = Generator(enc_dim=self.hparams.ngf, dec_dim=self.hparams.ngf,
                                   n_attrs=self.hparams.n_attrs, img_size=256)
            self.net_g_inc = 1
        elif self.hparams.netG == 'descar':
            from models.DeScarGan.descargan import Generator
            print('use descargan discriminator')
            self.net_g = Generator(n_channels=3)  ## i am using 32!
            self.net_g_inc = 2
        else:
            self.net_g = define_G(input_nc=self.hparams.input_nc, output_nc=self.hparams.output_nc,
                                  ngf=self.hparams.ngf, netG=self.hparams.netG,
                                  norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[])
            self.net_g_inc = 0
        # DISCRIMINATOR
        # patchgan
        if (self.hparams.netD).startswith('patchgan'):
            from models.cyclegan.models import Discriminator
            self.net_d = Discriminator(input_shape=(6, 256, 256), patch=(self.hparams.netD).split('_')[-1])
        elif self.hparams.netD == 'sagan':
            from models.sagan.sagan import Discriminator
            print('use sagan discriminator')
            self.net_d = Discriminator(image_size=64)
        elif self.hparams.netD == 'acgan':
            from models.acgan import Discriminator
            print('use acgan discriminator')
            self.net_d = Discriminator(img_shape=(6, 256, 256), n_classes=2)
        elif self.hparams.netD == 'attgan':
            from models.AttGAN.attgan import Discriminators
            print('use attgan discriminator')
            self.net_d = Discriminators(img_size=256, cls=2)
        elif self.hparams.netD == 'descar':
            from models.DeScarGan.descargan import Discriminator
            print('use descargan discriminator')
            self.net_d = Discriminator()
        # original pix2pix, the size of patchgan is strange, just use for pixel-D
        else:
            self.net_d = define_D(input_nc=self.hparams.output_nc * 2, ndf=64, netD=self.hparams.netD)

        # INIT
        self.net_g = self.net_g.apply(_weights_init)
        if self.hparams.netD == 'sagan':
            print('oo')
        else:
            self.net_d = self.net_d.apply(_weights_init)

        self.classifier = MRPretrained()
        self.CELoss = CrossEntropyLoss()

        self.seg_model = torch.load(os.environ.get('model_seg')).cuda()

        [self.optimizer_d, self.optimizer_g], [] = self.configure_optimizers()
        self.net_g_scheduler = get_scheduler(self.optimizer_g, self.hparams)
        self.net_d_scheduler = get_scheduler(self.optimizer_d, self.hparams)

        self.dir_checkpoints = checkpoints

        self.criterionL1 = nn.L1Loss().cuda()
        self.criterionL1_weighted = nn.L1Loss(reduction='none').cuda()
        if self.hparams.gan_mode == 'vanilla':
            self.criterionGAN = nn.BCEWithLogitsLoss()
        else:
            self.criterionGAN = GANLoss(self.hparams.gan_mode).cuda()

        self.segLoss = SegmentationCrossEntropyLoss()

        # final hparams
        self.hparams.update(vars(self.hparams))

        print(print_num_of_parameters(self.net_g))

    def configure_optimizers(self):
        self.optimizer_g = optim.Adam(list(self.net_g.parameters()) + list(self.classifier.parameters()), lr=self.hparams.lr, betas=(self.hparams.beta1, 0.999))
        self.optimizer_d = optim.Adam(self.net_d.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, 0.999))

        return [self.optimizer_d, self.optimizer_g], []

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        parser.add_argument("--n_attrs", type=int, default=1)
        return parent_parser

    def add_loss_adv(self, a, b, loss, coeff, truth, log=None, stacked=False):
        if stacked:
            fake_in = torch.cat((a, b), 1)
        else:
            fake_in = torch.cat((a, a), 1)
        disc_logits = self.net_d(fake_in)[0]
        if truth:
            adv = self.criterionGAN(disc_logits, torch.ones_like(disc_logits))
        else:
            adv = self.criterionGAN(disc_logits, torch.zeros_like(disc_logits))
        loss = loss + coeff * adv
        if log is not None:
            self.log(log, coeff * adv, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def add_loss_L1(self, a, b, loss, coeff, log=None):
        l1 = self.criterionL1(a, b)
        loss = loss + coeff * l1
        if log is not None:
            self.log(log, coeff * l1, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def add_loss_L1_weighted(self, a, b, loss, coeff, log=None, weight=None):
        l1 = self.criterionL1_weighted(a, b)
        l1 = l1.mean(1)
        l1 = torch.mul(l1, weight).mean()
        loss = loss + coeff * l1
        if log is not None:
            self.log(log, coeff * l1, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def backward_g(self, inputs):
        self.net_g.zero_grad()
        loss_g = 0
        oriX = inputs[0]
        oriY = inputs[1]
        if self.hparams.bysubject:
            # torch.Size([B, 23, 3, 256, 256])
            (B, S, C, H, W) = oriX.shape
            oriX = oriX.view(B*S, C, H, W)
            oriY = oriY.view(B*S, C, H, W)
            BS = B * S
        else:
            (B, C, H, W) = oriX.shape
            BS = B

        # generating...
        imgX0 = self.net_g(oriX, a=torch.zeros(BS, self.net_g_inc).cuda())[0]
        imgX1 = self.net_g(oriX, a=torch.ones(BS, self.net_g_inc).cuda())[0]
        #imgX01 = self.net_g(imgX0, a=torch.ones(BS, self.net_g_inc).cuda())[0]  # cyc
        #imgY10 = self.net_g(imgY1, a=torch.zeros(BS, self.net_g_inc).cuda())[0]  # cyc

        # ADV(X0)+
        loss_g = self.add_loss_adv(a=imgX0, b=None, loss=loss_g, coeff=1, truth=True, stacked=False)

        # L1(X0, Y)
        loss_g = self.add_loss_L1(a=imgX0, b=oriY, loss=loss_g, coeff=self.hparams.lamb)

        # L1(X1, X)
        loss_g = self.add_loss_L1(a=imgX1, b=oriX, loss=loss_g, coeff=self.hparams.lamb * 0.1)

        # ADV(X1)+
        loss_g = self.add_loss_adv(a=imgX1, b=None, loss=loss_g, coeff=1, truth=True, stacked=False)
        return loss_g

    def backward_d(self, inputs):
        loss_d = 0
        self.net_d.zero_grad()
        oriX = inputs[0]
        oriY = inputs[1]

        if self.hparams.bysubject:
            (B, S, C, H, W) = oriX.shape
            oriX = oriX.view(B*S, C, H, W)
            oriY = oriY.view(B*S, C, H, W)
            BS = B * S
        else:
            (B, C, H, W) = oriX.shape
            BS = B

        if self.net_g_inc > 0:
            imgX0 = self.net_g(oriX, torch.zeros(BS, self.net_g_inc).cuda())[0].detach()
            imgX1 = self.net_g(oriX, torch.ones(BS, self.net_g_inc).cuda())[0].detach()
        else:
            imgX0 = self.net_g(oriX)[0].detach()

        ######
        # ADV(X0)-
        loss_d = self.add_loss_adv(a=imgX0, b=None, loss=loss_d, coeff=0.25, truth=False, stacked=False)

        # ADV(Y)+
        loss_d = self.add_loss_adv(a=oriY, b=None, loss=loss_d, coeff=0.5, truth=True)

        # ADV(X1)-
        loss_d = self.add_loss_adv(a=imgX1, b=None, loss=loss_d, coeff=0.25, truth=False, stacked=False)

        self.log('loss_d', loss_d, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss_d

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = batch
        if optimizer_idx == 0:
            #self.net_d.zero_grad()
            #for param in self.net_d.parameters():
            #    param.requires_grad = True
            loss_d = self.backward_d(inputs)
            return loss_d

        if optimizer_idx == 1:
            #self.net_g.zero_grad()
            #for param in self.net_d.parameters():
            #    param.requires_grad = False
            loss_g = self.backward_g(inputs)
            self.log('loss_g', loss_g, on_step=False, on_epoch=True,
                     prog_bar=True, logger=True, sync_dist=True)
            return loss_g

    def training_epoch_end(self, outputs):
        hparams = self.hparams
        self.net_g_scheduler.step()
        self.net_d_scheduler.step()
        # checkpoint
        dir_checkpoints = self.dir_checkpoints
        if self.epoch % 10 == 0:
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
