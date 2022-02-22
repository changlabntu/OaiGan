from engine.base import BaseModel
import copy
import torch


class GAN(BaseModel):
    """
    There is a lot of patterned noise and other failures when using lightning
    """
    def __init__(self, hparams, train_loader, test_loader, checkpoints):
        BaseModel.__init__(self, hparams, train_loader, test_loader, checkpoints)

        self.net_gXY = self.net_g
        self.net_gYX = copy.deepcopy(self.net_g)

        self.net_dX = self.net_d
        self.net_dY = copy.deepcopy(self.net_d)

        # save model names
        self.netg_names = {'net_gXY': 'netGXY', 'net_gYX': 'netGYX'}
        self.netd_names = {'net_dX': 'netDX', 'net_dY': 'netDY'}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        # coefficient for the identify loss
        parser.add_argument("--lambI", type=int, default=0.5)
        #parser.add_argument("--lambD", type=int, default=100)
        return parent_parser

    def generation(self):
        self.oriX = self.batch[0]
        self.oriY = self.batch[2]
        self.oriZ = self.batch[1]

        self.imgXY0 = self.net_gXY(self.oriX, a=torch.zeros(self.oriX.shape[0], self.net_g_inc).cuda())[0]
        self.imgXY1 = self.net_gXY(self.oriX, a=torch.ones(self.oriX.shape[0], self.net_g_inc).cuda())[0]
        self.imgYX0 = self.net_gYX(self.oriY, a=torch.zeros(self.oriX.shape[0], self.net_g_inc).cuda())[0]
        #self.imgYX1 = self.net_gYX(oriY, a=torch.ones(oriX.shape[0], self.net_g_inc).cuda())[0]

        self.imgXYX = self.net_gYX(self.imgXY0, a=torch.zeros(self.oriX.shape[0], self.net_g_inc).cuda())[0]
        self.imgYXY = self.net_gXY(self.imgYX0, a=torch.zeros(self.oriX.shape[0], self.net_g_inc).cuda())[0]

        if self.hparams.lambI > 0:
            self.idt_X = self.net_gYX(self.oriX, a=torch.zeros(self.oriX.shape[0], self.net_g_inc).cuda())[0]
            self.idt_Y = self.net_gXY(self.oriY, a=torch.zeros(self.oriX.shape[0], self.net_g_inc).cuda())[0]

    def backward_g(self, inputs):
        loss_g = 0
        # ADV(XY)+
        loss_g = self.add_loss_adv(a=self.imgXY0, b=None, net_d=self.net_dY, loss=loss_g, coeff=1, truth=True, stacked=False)
        # ADV(YX)+
        loss_g = self.add_loss_adv(a=self.imgYX0, b=None, net_d=self.net_dX, loss=loss_g, coeff=1, truth=True, stacked=False)

        # Cyclic(XYX, X)
        loss_g = self.add_loss_L1(a=self.imgXYX, b=self.oriX, loss=loss_g, coeff=self.hparams.lamb)
        # Cyclic(YXY, Y)
        loss_g = self.add_loss_L1(a=self.imgYXY, b=self.oriY, loss=loss_g, coeff=self.hparams.lamb)

        # Identity(idt_X, X)
        if self.hparams.lambI > 0:
            loss_g = self.add_loss_L1(a=self.idt_X, b=self.oriX, loss=loss_g, coeff=self.hparams.lambI)
            # Identity(idt_Y, Y)
            loss_g = self.add_loss_L1(a=self.idt_Y, b=self.oriY, loss=loss_g, coeff=self.hparams.lambI)

        #
        loss_g = self.add_loss_L1(a=self.imgXY1, b=self.oriZ, loss=loss_g, coeff=10)

        return loss_g

    def backward_d(self, inputs):
        loss_d = 0
        # ADV(XY)-
        loss_d = self.add_loss_adv(a=self.imgXY0, net_d=self.net_dY, loss=loss_d, coeff=1, truth=False, stacked=False)

        # ADV(YX)-
        loss_d = self.add_loss_adv(a=self.imgYX0, net_d=self.net_dX, loss=loss_d, coeff=1, truth=False, stacked=False)

        # ADV(Y)+
        loss_d = self.add_loss_adv(a=self.oriY, net_d=self.net_dY, loss=loss_d, coeff=1, truth=True, stacked=False)

        # ADV(X)+
        loss_d = self.add_loss_adv(a=self.oriX, net_d=self.net_dX, loss=loss_d, coeff=1, truth=True, stacked=False)

        return loss_d

# CUDA_VISIBLE_DEVICES=0 python train.py --dataset FlyZ -b 16 --prj WnWpD0 --direction zyweak_zyori%xyweak --resize 286 --engine cycleganDual --lamb 10 --netG attgan