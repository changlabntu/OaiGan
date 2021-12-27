from engine.base import BaseModel


class Pix2PixModel(BaseModel):
    """
    There is a lot of patterned noise and other failures when using lightning
    """
    def __init__(self, hparams, train_loader, test_loader, checkpoints):
        BaseModel.__init__(self, hparams, train_loader, test_loader, checkpoints)
        print('using pix2pix.py')

    def generation(self):
        oriX = self.oriX
        oriY = self.oriY

        self.imgXY = self.net_gXY(oriX)[0]
        self.imgYX = self.net_gYX(oriY)[0]

        self.imgXYX = self.netg_YX(self.imgXY)[0]
        self.imgYXY = self.netg_XY(self.imgYX)[0]

    def backward_g(self, inputs):
        loss_g = 0
        # ADV(XY, Y)+
        loss_g = self.add_loss_adv(a=self.imgXY, b=self.oriY, loss=loss_g, coeff=1, truth=True, stacked=False)
        # ADV(YX, X)+
        loss_g = self.add_loss_adv(a=self.imgYX, b=self.oriX, loss=loss_g, coeff=1, truth=True, stacked=False)

        # Cyclic(XYX, X)+
        loss_g = self.add_loss_adv(a=self.imgXYX, b=self.oriX, loss=loss_g, coeff=1, truth=True, stacked=True)
        # Cyclic(YXY, Y)+
        loss_g = self.add_loss_adv(a=self.imgYXY, b=self.oriY, loss=loss_g, coeff=1, truth=True, stacked=True)

        # Identity(XYX, X)
        loss_g = self.add_loss_L1(a=self.imgXYX, b=self.oriX, loss=loss_g, coeff=self.hparams.lamb)
        # Identity(YXY, Y)
        loss_g = self.add_loss_L1(a=self.imgYXY, b=self.oriY, loss=loss_g, coeff=self.hparams.lamb)

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

