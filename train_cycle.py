from __future__ import print_function
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.make_config import load_config
path = load_config('config.ini', 'Path')


# Training settings
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--dataset', type=str, default='facades')
parser.add_argument('--prj', type=str, default='', help='name of the project')
parser.add_argument('-b', dest='batch_size', type=int, default=1, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--direction', type=str, default='a_b', help='a2b or b2a')
parser.add_argument('--gan_mode', type=str, default='lsgan', help='gan mode')
parser.add_argument('--netG', type=str, default='unet_256', help='netG model')
parser.add_argument('--netD', type=str, default='basic', help='netD model')
parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
parser.add_argument('--epoch_count', type=int, default=0, help='the starting epoch count')
parser.add_argument('--epoch_load', type=int, default=0, help='to load checkpoint form the epoch count')
parser.add_argument('--n_epochs', type=int, default=500, help='# of iter at starting learning rate')
parser.add_argument('--n_epochs_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate f -or adam')
parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--lambda_cyc', type=float, default=10, help='weight on L1 term in cyclic loss')
parser.add_argument('--lambda_id', type=float, default=5, help='weight on L1 term in objective')
parser.add_argument('--legacy', action='store_true', dest='legacy', default=False, help='legacy pytorch')
parser.add_argument('--mode', type=str, default='dummy')
parser.add_argument('--port', type=str, default='dummy')
opt = parser.parse_args()#(args=[])
opt.prj = opt.dataset + '_' + opt.prj
print(opt)

#cudnn.benchmark = True

#torch.manual_seed(opt.seed)
#torch.cuda.manual_seed(opt.seed)

#  Dataset
if opt.dataset == 'dess':
    from dataloader.data_dess_segmentation import DatasetDessSegmentation
    train_set = DatasetDessSegmentation(mode='train', direction=opt.direction)
    test_set = DatasetDessSegmentation(mode='val', direction=opt.direction)
else:
    from dataloader.data import get_training_set, get_test_set
    root_path = path['dataset']
    train_set = get_training_set(root_path + opt.dataset, opt.direction, mode='train')
    test_set = get_test_set(root_path + opt.dataset, opt.direction, mode='train')

print('training set length: ' + str(len(train_set)))
print('testing set length: ' + str(len(test_set)))

train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.test_batch_size, shuffle=False)

#  Model

from models.cyclegan.cycleganln import CycleGanModel
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
net = CycleGanModel(hparams=opt, dir_checkpoints=path['checkpoints'])
print(net.hparams)
tb_logger = pl_loggers.TensorBoardLogger(path['logs'])
trainer = pl.Trainer(gpus=[0],# distributed_backend='ddp',
                     max_epochs=opt.n_epochs, progress_bar_refresh_rate=20, logger=tb_logger)
trainer.fit(net, train_loader, test_loader)

