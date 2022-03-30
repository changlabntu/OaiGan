from __future__ import print_function
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
import os, shutil, copy
from dotenv import load_dotenv
from utils.make_config import *

def prepare_log(opt):
    """
    finalize arguments, creat a folder for logging, save argument in json
    """
    opt.not_tracking_hparams = ['mode', 'port', 'epoch_load', 'legacy', 'threads', 'test_batch_size']
    os.makedirs(os.environ.get('LOGS') + opt.dataset + '/', exist_ok=True)
    os.makedirs(os.environ.get('LOGS') + opt.dataset + '/' + opt.prj + '/', exist_ok=True)
    save_json(opt, os.environ.get('LOGS') + opt.dataset + '/' + opt.prj + '/' + '0.json')
    shutil.copy('engine/' + opt.engine + '.py', os.environ.get('LOGS') + opt.dataset + '/' + opt.prj + '/' + opt.engine + '.py')
    return opt

# Arguments
parser = argparse.ArgumentParser()#add_help=False)
# Env
parser.add_argument('--jsn', type=str, default='0', help='name of ini file')
parser.add_argument('--env', type=str, default=None, help='environment_to_use')
# Project name
parser.add_argument('--prj', type=str, default='', help='name of the project')
parser.add_argument('--engine', dest='engine', type=str, default='mydcgan', help='use which engine')
# Data
parser.add_argument('--dataset', type=str, default='pain')
parser.add_argument('--bysubject', action='store_true', dest='bysubject', default=False)
parser.add_argument('--direction', type=str, default='a_b', help='a2b or b2a')
parser.add_argument('--flip', action='store_true', dest='flip', default=False, help='image flip left right')
parser.add_argument('--resize', type=int, default=0, help='size for resizing before cropping, 0 for no resizing')
parser.add_argument('--cropsize', type=int, default=256, help='size for cropping, 0 for no crop')
parser.add_argument('--n01', action='store_true', dest='n01', default=False, help='normalized to 0~1 instead of -1~1')
#parser.add_argument('--gray', action='store_true', dest='gray', default=False, help='only use 1 channel')
# Model
parser.add_argument('--gan_mode', type=str, default='vanilla', help='gan mode')
parser.add_argument('--netG', type=str, default='unet_256', help='netG model')
parser.add_argument('--norm', type=str, default='batch', help='normalization in generator')
parser.add_argument('--mc', action='store_true', dest='mc', default=False, help='monte carlo dropout for pix2pix generator')
parser.add_argument('--netD', type=str, default='patch_16', help='netD model')
parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
parser.add_argument("--n_attrs", type=int, default=1)
parser.add_argument('--final', type=str, dest='final', default='tanh', help='activation of final layer')
parser.add_argument('--cmb', dest='cmb', default=None, help='method to combine the outputs to the original')
# Training
parser.add_argument('-b', dest='batch_size', type=int, default=1, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--n_epochs', type=int, default=201, help='# of iter at starting learning rate')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate f -or adam')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--epoch_count', type=int, default=0, help='the starting epoch count')
parser.add_argument('--epoch_load', type=int, default=0, help='to load checkpoint form the epoch count')
parser.add_argument('--n_epochs_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
# Loss
parser.add_argument('--lamb', type=int, default=100, help='weight on L1 term in objective')
# Misc
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--legacy', action='store_true', dest='legacy', default=False, help='legacy pytorch')
parser.add_argument('--mode', type=str, default='dummy')
parser.add_argument('--port', type=str, default='dummy')

with open('engine/' + parser.parse_args().jsn + '.json', 'rt') as f:
    t_args = argparse.Namespace()
    t_args.__dict__.update(json.load(f))
    args = parser.parse_args(namespace=t_args)

if 0:
    # Model-specific Arguments
    engine = parser.parse_known_args()[0].engine
    GAN = getattr(__import__('engine.' + engine), engine).GAN
    parser = GAN.add_model_specific_args(parser)
    opt = parser.parse_args()

    # environment file
    if opt.env is not None:
        load_dotenv('.' + opt.env)
    else:
        load_dotenv('.env')

    # Finalize Arguments and create files for logging
    opt = prepare_log(opt)

    #  Define Dataset Class
    from dataloader.data_multi import PairedData3D as Dataset

    # Load Dataset and DataLoader
    train_set = Dataset(root=os.environ.get('DATASET') + opt.dataset + '/train/',
                        path=opt.direction,
                        opt=opt, mode='train')
    print(len(train_set))
    train_set = Dataset(root=os.environ.get('DATASET') + opt.dataset + '/train/',
                        path=opt.direction,
                        opt=opt, mode='train', index=range(len(train_set) // 10 * 3, len(train_set)))
    print(len(train_set))

    train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True)

    #  Pytorch Lightning Module
    import pytorch_lightning as pl
    from pytorch_lightning import loggers as pl_loggers

    logger = pl_loggers.TensorBoardLogger(os.environ.get('LOGS') + opt.dataset + '/', name=opt.prj)
    checkpoints = os.path.join(os.environ.get('LOGS'), opt.dataset, opt.prj, 'checkpoints')
    os.makedirs(checkpoints, exist_ok=True)
    net = GAN(hparams=opt, train_loader=None,
              test_loader=None, checkpoints=checkpoints)
    trainer = pl.Trainer(gpus=[0],  # distributed_backend='ddp',
                         max_epochs=opt.n_epochs, progress_bar_refresh_rate=20, logger=logger)
    trainer.fit(net, train_loader)#, test_loader)  # test loader not used during training


# Example Usage
# CUDA_VISIBLE_DEVICES=1 python train.py --dataset TSE_DESS -b 16 --prj VryCycle --direction a_b --resize 286 --engine cyclegan --lamb 10 --unpaired
# CUDA_VISIBLE_DEVICES=1 python train.py --dataset pain -b 16 --prj VryNS4B --direction aregis1_b --resize 286 --engine NS4 --netG attgan
# CUDA_VISIBLE_DEVICES=0 python train.py --dataset FlyZ -b 16 --prj WpWn286B --direction xyweak%zyweak --resize 286 --engine cyclegan --lamb 10
# CUDA_VISIBLE_DEVICES=1 python train.py --dataset FlyZ -b 16 --prj WpOp256Mask --direction xyweak_xyorisb --resize 256 --engine pix2pixNS


# CUDA_VISIBLE_DEVICES=0 python train.py --dataset womac3 -b 1 --prj bysubjectright/descar2/GDdescars --direction areg_b --cropsize 256 --engine descar2 --netG descars --netD descar --n01 --final sigmoid --cmb mul --bysubject