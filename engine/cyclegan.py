from engine.base import BaseModel
import copy


class Pix2PixModel(BaseModel):
    """
    There is a lot of patterned noise and other failures when using lightning
    """
    def __init__(self, hparams, train_loader, test_loader, checkpoints):
        BaseModel.__init__(self, hparams, train_loader, test_loader, checkpoints)
        print('using pix2pix.py')

        self.net_gXY = self.net_g
        self.net_gYX = copy.deepcopy(self.net_g)

        self.net_dX = self.net_d
        self.net_dY = copy.deepcopy(self.net_d)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        # coefficient for the identify loss
        parser.add_argument("--lambI", type=int, default=0)
        return parent_parser

    def generation(self):
        oriX = self.oriX
        oriY = self.oriY

        self.imgXY = self.net_gXY(oriX)[0]
        self.imgYX = self.net_gYX(oriY)[0]

        self.imgXYX = self.netg_YX(self.imgXY)[0]
        self.imgYXY = self.netg_XY(self.imgYX)[0]

        if self.hparams.lamb > 0:
            self.idt_X = self.net_gYX(oriX)[0]
            self.idt_Y = self.net_gXY(oriY)[0]

    def backward_g(self, inputs):
        loss_g = 0
        # ADV(XY)+
        loss_g = self.add_loss_adv(a=self.imgXY, b=None, net_d=self.net_dY, loss=loss_g, coeff=1, truth=True, stacked=False)
        # ADV(YX)+
        loss_g = self.add_loss_adv(a=self.imgYX, b=None, net_d=self.net_dX, loss=loss_g, coeff=1, truth=True, stacked=False)

        # Cyclic(XYX, X)
        loss_g = self.add_loss_L1(a=self.imgXYX, b=self.oriX, loss=loss_g, coeff=self.hparams.lamb)
        # Cyclic(YXY, Y)
        loss_g = self.add_loss_L1(a=self.imgYXY, b=self.oriY, loss=loss_g, coeff=self.hparams.lamb)

        # Identity(idt_X, X)
        if self.hparams.lamb > 0:
            loss_g = self.add_loss_L1(a=self.idt_X, b=self.oriX, loss=loss_g, coeff=self.hparams.lamb)
            # Identity(idt_Y, Y)
            loss_g = self.add_loss_L1(a=self.idt_Y, b=self.oriY, loss=loss_g, coeff=self.hparams.lamb)

        return loss_g

    def backward_d(self, inputs):
        loss_d = 0
        # ADV(XY, Y)+
        loss_d = self.add_loss_adv(a=self.imgXY, b=self.oriY, loss=loss_d, coeff=1, truth=False, stacked=False)
        # ADV(YX, X)+
        loss_d = self.add_loss_adv(a=self.imgYX, b=self.oriX, loss=loss_d, coeff=1, truth=False, stacked=False)

        # ADV(XY, Y)+
        loss_d = self.add_loss_adv(a=self.oriY, b=self.oriY, loss=loss_d, coeff=1, truth=False, stacked=False)
        # ADV(YX, X)+
        loss_d = self.add_loss_adv(a=self.oriX, b=self.oriX, loss=loss_d, coeff=1, truth=False, stacked=False)

        # Cyclic(XYX, X)+
        loss_d = self.add_loss_adv(a=self.imgXYX, b=self.oriX, loss=loss_d, coeff=1, truth=False, stacked=True)
        # Cyclic(YXY, Y)+
        loss_d = self.add_loss_adv(a=self.imgYXY, b=self.oriY, loss=loss_d, coeff=1, truth=False, stacked=True)
        return loss_d

