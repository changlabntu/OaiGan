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
from utils.metrics_segmentation import SegmentationCrossEntropyLoss


class DistanceGAN:
    def __init__(self):
        self.init = None

    def distance(self, A, B):
        return torch.mean(torch.abs(A - B))

    def get_individual_distance_loss(self, A_i, A_j, AB_i, AB_j,
                                     B_i, B_j, BA_i, BA_j):

        distance_in_A = self.distance(A_i, A_j)
        distance_in_AB = self.distance(AB_i, AB_j)
        distance_in_B = self.distance(B_i, B_j)
        distance_in_BA = self.distance(BA_i, BA_j)

        if self.normalize_distances:
            distance_in_A = (distance_in_A - self.expectation_A) / self.std_A
            distance_in_AB = (distance_in_AB - self.expectation_B) / self.std_B
            distance_in_B = (distance_in_B - self.expectation_B) / self.std_B
            distance_in_BA = (distance_in_BA - self.expectation_A) / self.std_A

        return torch.abs(distance_in_A - distance_in_AB), torch.abs(distance_in_B - distance_in_BA)

    def get_self_distances(self, A, B, AB, BA):

        A_half_1, A_half_2 = torch.chunk(A, 2, dim=2)
        B_half_1, B_half_2 = torch.chunk(B, 2, dim=2)
        AB_half_1, AB_half_2 = torch.chunk(AB, 2, dim=2)
        BA_half_1, BA_half_2 = torch.chunk(BA, 2, dim=2)

        l_distance_A, l_distance_B = \
            self.get_individual_distance_loss(A_half_1, A_half_2,
                                              AB_half_1, AB_half_2,
                                              B_half_1, B_half_2,
                                              BA_half_1, BA_half_2)

        return l_distance_A, l_distance_B

    def get_distance_losses(self, A, B, AB, BA):

        As = torch.split(A, 1)
        Bs = torch.split(B, 1)
        ABs = torch.split(AB, 1)
        BAs = torch.split(BA, 1)

        loss_distance_A = 0.0
        loss_distance_B = 0.0
        num_pairs = 0
        min_length = min(len(As), len(Bs))

        for i in range(min_length - 1):
            for j in range(i + 1, min_length):
                num_pairs += 1
                loss_distance_A_ij, loss_distance_B_ij = \
                    self.get_individual_distance_loss(As[i], As[j],
                                                      ABs[i], ABs[j],
                                                      Bs[i], Bs[j],
                                                      BAs[i], BAs[j])

                loss_distance_A += loss_distance_A_ij
                loss_distance_B += loss_distance_B_ij

        loss_distance_A = loss_distance_A / num_pairs
        loss_distance_B = loss_distance_B / num_pairs

        return loss_distance_A, loss_distance_B



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
        # additional models

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
        if hparams.netD == 'patchgan':
            self.net_d = Discriminator(input_shape=(6, 256, 256))
        else:
            self.net_d = define_D(input_nc=hparams.output_nc * 2, ndf=64, netD=hparams.netD)

        self.seg_model = torch.load(os.environ.get('model_seg')).cuda()

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

        self.segLoss = SegmentationCrossEntropyLoss()

        self.normalize_distances = False

    def distance(self, A, B):
        return torch.mean(torch.abs(A - B))

    def get_individual_distance_loss(self, A_i, A_j, AB_i, AB_j,
                                     B_i, B_j, BA_i, BA_j):

        distance_in_A = self.distance(A_i, A_j)
        distance_in_AB = self.distance(AB_i, AB_j)
        distance_in_B = self.distance(B_i, B_j)
        distance_in_BA = self.distance(BA_i, BA_j)

        if self.normalize_distances:
            distance_in_A = (distance_in_A - self.expectation_A) / self.std_A
            distance_in_AB = (distance_in_AB - self.expectation_B) / self.std_B
            distance_in_B = (distance_in_B - self.expectation_B) / self.std_B
            distance_in_BA = (distance_in_BA - self.expectation_A) / self.std_A

        return torch.abs(distance_in_A - distance_in_AB), torch.abs(distance_in_B - distance_in_BA)

    def get_self_distances(self, A, B, AB, BA):

        A_half_1, A_half_2 = torch.chunk(A, 2, dim=2)
        B_half_1, B_half_2 = torch.chunk(B, 2, dim=2)
        AB_half_1, AB_half_2 = torch.chunk(AB, 2, dim=2)
        BA_half_1, BA_half_2 = torch.chunk(BA, 2, dim=2)

        l_distance_A, l_distance_B = \
            self.get_individual_distance_loss(A_half_1, A_half_2,
                                              AB_half_1, AB_half_2,
                                              B_half_1, B_half_2,
                                              BA_half_1, BA_half_2)

        return l_distance_A, l_distance_B

    def get_distance_losses(self, A, B, AB, BA):

        As = torch.split(A, 1)
        Bs = torch.split(B, 1)
        ABs = torch.split(AB, 1)
        BAs = torch.split(BA, 1)

        loss_distance_A = 0.0
        loss_distance_B = 0.0
        num_pairs = 0
        min_length = min(len(As), len(Bs))

        for i in range(min_length - 1):
            for j in range(i + 1, min_length):
                num_pairs += 1
                loss_distance_A_ij, loss_distance_B_ij = \
                    self.get_individual_distance_loss(As[i], As[j],
                                                      ABs[i], ABs[j],
                                                      Bs[i], Bs[j],
                                                      BAs[i], BAs[j])

                loss_distance_A += loss_distance_A_ij
                loss_distance_B += loss_distance_B_ij

        loss_distance_A = loss_distance_A / num_pairs
        loss_distance_B = loss_distance_B / num_pairs

        return loss_distance_A, loss_distance_B

    def configure_optimizers(self):
        self.optimizer_g = optim.Adam(self.net_g.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, 0.999))
        self.optimizer_d = optim.Adam(self.net_d.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, 0.999))

        return [self.optimizer_d, self.optimizer_g], []

    def backward_g(self, inputs):
        conditioned_images = inputs[0]
        real_images = inputs[1]

        self.net_g.zero_grad()
        fake_images = self.net_g(conditioned_images)
        disc_logits = self.net_d(torch.cat((fake_images, conditioned_images), 1))
        adversarial_loss = self.criterionGAN(disc_logits, torch.ones_like(disc_logits))

        # calculate reconstruction loss
        recon_loss = self.criterionL1(fake_images, real_images)
        loss_g = adversarial_loss + self.hparams.lamb * recon_loss

        # segmentation loss
        prob = self.seg_model(fake_images)[0]   # (16, 3, 256, 256)
        if 0: # use the segmented dess mask (bone only)
            seg_images = inputs[2]
            seg = seg_images.type(torch.LongTensor).cuda()
        elif 0: # use the segment model for dess
            seg = self.seg_model(real_images)[0]
            seg = torch.argmax(seg, 1)
        elif 1:
            seg = self.seg_model(conditioned_images)[0]
            seg = torch.argmax(seg, 1)
        loss_seg, _ = self.segLoss((prob, ), (seg, ))
        self.log('loss_seg', loss_seg, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)

        loss_g = loss_g + loss_seg * self.hparams.lseg

        # target domain identity loss
        if 0:
            fake_images_b = self.net_g(real_images)
            recon_loss_b = nn.MSELoss()(fake_images_b, real_images)
            loss_g = loss_g + adversarial_loss + 1 * self.hparams.lamb * recon_loss_b

        # distance loss
        A = conditioned_images
        B = real_images
        AB = self.net_g(A)
        BB = self.net_g(B)
        self.loss_distance_A, self.loss_distance_B = self.get_distance_losses(A, B, AB, BB)
        self.log('distA', self.loss_distance_A, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)
        self.log('distB', self.loss_distance_B, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)

        loss_g = loss_g + self.loss_distance_A + self.loss_distance_B

        return loss_g

    def backward_d(self, inputs):
        conditioned_images = inputs[0]
        real_images = inputs[1]

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
