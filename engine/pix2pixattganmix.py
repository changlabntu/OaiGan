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
        if self.hparams.netG == 'attgan':
            self.GZ = True
        else:
            self.GZ = False

        # GENERATOR
        if self.hparams.netG == 'attgan':
            from models.AttGAN.attgan import Generator
            print('use acgan discriminator')
            self.net_g = Generator(n_attrs=self.hparams.n_attrs, img_size=256)
        else:
            self.net_g = define_G(input_nc=self.hparams.input_nc, output_nc=self.hparams.output_nc, ngf=64, netG=self.hparams.netG,
                                  norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[])
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

        if self.hparams.lseg > 0:
            self.seg_model = torch.load(os.environ.get('model_seg')).cuda()

        [self.optimizer_d, self.optimizer_g], [] = self.configure_optimizers()
        self.net_g_scheduler = get_scheduler(self.optimizer_g, self.hparams)
        self.net_d_scheduler = get_scheduler(self.optimizer_d, self.hparams)

        self.dir_checkpoints = checkpoints

        self.criterionL1 = nn.L1Loss().cuda()
        if self.hparams.gan_mode == 'vanilla':
            self.criterionGAN = nn.BCEWithLogitsLoss()
        else:
            self.criterionGAN = GANLoss(self.hparams.gan_mode).cuda()

        self.segLoss = SegmentationCrossEntropyLoss()

        # final hparams
        self.hparams.update(vars(self.hparams))

    def configure_optimizers(self):
        self.optimizer_g = optim.Adam(list(self.net_g.parameters()) + list(self.classifier.parameters()), lr=self.hparams.lr, betas=(self.hparams.beta1, 0.999))
        self.optimizer_d = optim.Adam(self.net_d.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, 0.999))

        return [self.optimizer_d, self.optimizer_g], []

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        parser.add_argument("--n_attrs", type=int, default=1)
        return parent_parser

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

        # ADV(X, X0)+
        if self.GZ:
            imgX0 = self.net_g(oriX, a=torch.zeros(B * S, 1).cuda())[0]
        else:
            imgX0 = self.net_g(oriX)[0]
        fake_in = torch.cat((imgX0, oriX), 1)
        disc_logits = self.net_d(fake_in)[0]
        adv_XX0 = self.criterionGAN(disc_logits, torch.ones_like(disc_logits))
        loss_g = loss_g + 1 * adv_XX0

        # imgX1
        imgX1 = self.net_g(oriX, a=torch.ones(B * S, 1).cuda())[0]

        # MIX oriX and oriY
        dx = oriX.shape[2] // 2

        oriA = 1 * oriX
        oriB = 1 * oriY
        oriA[:, :, :, :dx] = oriX[:, :, :, :dx]
        oriA[:, :, :, dx:] = oriY[:, :, :, dx:]
        oriB[:, :, :, :dx] = oriY[:, :, :, :dx]
        oriB[:, :, :, dx:] = oriX[:, :, :, dx:]
        oriX = oriA
        oriY = oriB

        # ADV(X, X1)+
        fake_in = torch.cat((imgX1, oriX), 1)
        disc_logits = self.net_d(fake_in)[0]
        adv_XX1 = self.criterionGAN(disc_logits, torch.ones_like(disc_logits))
        loss_g = loss_g + 1 * adv_XX1

        # L1(Y, X0)
        recon_X0Y = self.criterionL1(imgX0, oriY)
        loss_g = loss_g + self.hparams.lamb * recon_X0Y

        self.log('loss_recon_a', recon_X0Y, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)

        #  L1(X, X1)
        recon_XX1 = self.criterionL1(imgX1, oriX)
        loss_g = loss_g + self.hparams.lamb * recon_XX1

        self.log('loss_recon_b', recon_XX1, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)

        if 0:
            #  L1(Y, Y0)
            imgY0 = self.net_g(oriY, a=torch.zeros(B * S, 1).cuda())[0]
            recon_Y0Y = self.criterionL1(imgY0, oriY)
            loss_g = loss_g + self.hparams.lamb * recon_Y0Y

            self.log('loss_recon_Y0Y', recon_Y0Y, on_step=False, on_epoch=True,
                     prog_bar=True, logger=True, sync_dist=True)

            #  L1(Y, Y1)
            imgY1 = self.net_g(oriY, a=torch.ones(B * S, 1).cuda())[0]
            recon_Y1Y = self.criterionL1(imgY1, oriY)
            loss_g = loss_g + self.hparams.lamb * recon_Y1Y

            self.log('loss_recon_Y1Y', recon_Y1Y, on_step=False, on_epoch=True,
                     prog_bar=True, logger=True, sync_dist=True)

        # target domain identity loss, not using it



        if 0:
            imgX0_b = self.net_g(imgY, a=torch.zeros(B*S, 1).cuda())[0]
            #recon_loss_b = nn.MSELoss()(imgX0_b, imgY)
            recon_loss_b = self.criterionL1(imgX0_b, imgY)
            loss_g = loss_g + self.hparams.lamb_b * recon_loss_b
            self.log('loss_recon_b', recon_loss_b, on_step=False, on_epoch=True,
                     prog_bar=True, logger=True, sync_dist=True)

        # Prior classification loss
        if 0:
            class_imgX, _ = self.classifier(imgX)
            class_imgY, _ = self.classifier(imgY)

            classfication_loss = self.CELoss(class_imgX,  torch.ones(1).type(torch.LongTensor).cuda())[0] +\
                                 self.CELoss(class_imgY, torch.zeros(1).type(torch.LongTensor).cuda())[0]

        if 0:
            class_out, _ = self.classifier((imgX, imgY))
            pred = torch.argmax(class_out, 1)
            classfication_loss = self.CELoss(class_out, torch.ones(1).type(torch.LongTensor).cuda())[0]
            loss_g = loss_g + 10 * classfication_loss
            self.log('loss_classify', classfication_loss, on_step=False, on_epoch=True,
                     prog_bar=True, logger=True, sync_dist=True)

            self.log('pred', pred, on_step=False, on_epoch=True,
                     prog_bar=True, logger=True, sync_dist=True)

        return loss_g

    def backward_d(self, inputs):
        self.net_d.zero_grad()
        oriX = inputs[0]
        oriY = inputs[1]
        if self.hparams.bysubject:
            (B, S, C, H, W) = oriX.shape
            oriX = oriX.view(B*S, C, H, W)
            oriY = oriY.view(B*S, C, H, W)
        if self.GZ:
            imgX0 = self.net_g(oriX, torch.zeros(B * S, 1).cuda())[0].detach()
            imgX1 = self.net_g(oriX, torch.ones(B * S, 1).cuda())[0].detach()
        else:
            imgX0 = self.net_g(oriX)[0].detach()

        # ADV(X0, X)-
        fake_in = torch.cat((imgX0, oriX), 1)
        fake_logits = self.net_d(fake_in)[0]
        fake_loss0 = self.criterionGAN(fake_logits, torch.zeros_like(fake_logits))

        # ADV(Y, X)+
        real_in = torch.cat((oriY, oriX), 1)
        real_logits = self.net_d(real_in)[0]
        real_loss = self.criterionGAN(real_logits, torch.ones_like(real_logits))

        if self.GZ:
            # ADV(X1, X)-
            fake_in = torch.cat((imgX1, oriX), 1)
            fake_logits = self.net_d(fake_in)[0]
            fake_loss1 = self.criterionGAN(fake_logits, torch.zeros_like(fake_logits))

        # Combined D loss
        if self.GZ:
            loss_d = real_loss * 0.5 + fake_loss0 * 0.25 + fake_loss1 * 0.25
        else:
            loss_d = real_loss * 0.5 + fake_loss0 * 0.5
        return loss_d

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = batch
        if optimizer_idx == 0:
            #self.net_d.zero_grad()
            #for param in self.net_d.parameters():
            #    param.requires_grad = True
            loss_d = self.backward_d(inputs)
            self.log('loss_d', loss_d, on_step=False, on_epoch=True,
                     prog_bar=True, logger=True, sync_dist=True)
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
