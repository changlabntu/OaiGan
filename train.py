from __future__ import print_function
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from dotenv import load_dotenv
load_dotenv('.env')

parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
# Data
parser.add_argument('--dataset', type=str, default='pain')
parser.add_argument('--bysubject', action='store_true', dest='bysubject', default=False)
parser.add_argument('--prj', type=str, default='', help='name of the project')
parser.add_argument('--direction', type=str, default='a_b', help='a2b or b2a')
parser.add_argument('--flip', action='store_true', dest='flip', default=False, help='image flip left right')
# Model
parser.add_argument('--gan_mode', type=str, default='vanilla', help='gan mode')
parser.add_argument('--netG', type=str, default='unet_256', help='netG model')
parser.add_argument('--netD', type=str, default='patchgan', help='netD model')
parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
# Training
parser.add_argument('-b', dest='batch_size', type=int, default=1, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--epoch_count', type=int, default=0, help='the starting epoch count')
parser.add_argument('--epoch_load', type=int, default=0, help='to load checkpoint form the epoch count')
parser.add_argument('--n_epochs', type=int, default=300+1, help='# of iter at starting learning rate')
parser.add_argument('--n_epochs_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate f -or adam')
parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--lamb', type=int, default=100, help='weight on L1 term in objective')
parser.add_argument('--lseg', type=int, default=0, help='weight on segmentation loss in objective')
parser.add_argument('--legacy', action='store_true', dest='legacy', default=False, help='legacy pytorch')
parser.add_argument('--mode', type=str, default='dummy')
parser.add_argument('--port', type=str, default='dummy')
opt = parser.parse_args()
opt.prj = opt.dataset + '_' + opt.prj
print(opt)

#cudnn.benchmark = True
#torch.manual_seed(opt.seed)
#torch.cuda.manual_seed(opt.seed)

#  Dataset
if opt.bysubject:
    from dataloader.data import DatasetFromFolderSubjects as Dataset
else:
    from dataloader.data import DatasetFromFolder as Dataset

train_set = Dataset(os.environ.get('DATASET') + opt.dataset + '/train/', opt, mode='train')
test_set = Dataset(os.environ.get('DATASET') + opt.dataset + '/test/', opt, mode='test')

print('training set length: ' + str(len(train_set)))
print('testing set length: ' + str(len(test_set)))

train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.test_batch_size, shuffle=False)

#  Model
if not opt.legacy:
    from models.pix2pix import Pix2PixModel
    import pytorch_lightning as pl
    from pytorch_lightning import loggers as pl_loggers
    logger = pl_loggers.TensorBoardLogger(os.environ.get('LOGS'))
    #from pytorch_lightning.loggers.neptune import NeptuneLogger
    #logger = NeptuneLogger(
    #    api_key=os.environ.get('NEPTUNE_API_TOKEN'),
    #    project_name="ntuchanglab/lightning-pix2pix",)
    net = Pix2PixModel(hparams=opt, train_loader=None,
                       test_loader=None, checkpoints=os.environ.get('CHECKPOINTS'))
    print(net.hparams)
    trainer = pl.Trainer(gpus=[0],  # distributed_backend='ddp',
                         max_epochs=opt.n_epochs, progress_bar_refresh_rate=20, logger=logger)
    trainer.fit(net, train_loader, test_loader)
else:
    from models.pix2pix import Pix2PixModel
    net = Pix2PixModel(hparams=opt, train_loader=train_loader,
                       test_loader=test_loader, checkpoints=os.environ.get('CHECKPOINTS'))
    net = net.cuda()
    net = nn.DataParallel(net)
    net.module.overall_loop()

# USAGE
# CUDA_VISIBLE_DEVICES=1 python train.py --dataset TSE_DESS -b 16 --prj TrySeg --direction a_b_bseg
# CUDA_VISIBLE_DEVICES=2 python train.py --dataset painfull -b 23 --prj patch16 --lseg 0 --direction aregis1_b
# CUDA_VISIBLE_DEVICES=1 python train.py --dataset pain -b 1 --prj bysubject --lseg 0 --direction aregis1_b --bysubject
# CUDA_VISIBLE_DEVICES=1 python train.py --dataset pain -b 16 --prj check --lseg 0 --direction aregis1_b
# CUDA_VISIBLE_DEVICES=0 python train.py --dataset pain -b 16 --prj Attv1_0 --lseg 0 --direction aregis1_b --netG Attv1_0

# CUDA_VISIBLE_DEVICES=1 python train.py --dataset pain -b 16 --prj AttUNet0 --lseg 0 --direction aregis1_b --netG AttUNet
