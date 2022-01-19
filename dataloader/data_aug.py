from os.path import join
import glob
import random

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
import numpy as np
import os
from skimage import io
from utils.data_utils import imagesc as show


def to_8bit(x):
    if type(x) == torch.Tensor:
        x = (x / x.max() * 255).numpy().astype(np.uint8)
    else:
        x = (x / x.max() * 255).astype(np.uint8)

    if len(x.shape) == 2:
        x = np.concatenate([np.expand_dims(x, 2)]*3, 2)
    return x


def imagesc(x, show=True, save=None):
    # switch
    if (len(x.shape) == 3) & (x.shape[0] == 3):
        x = np.transpose(x, (1, 2, 0))

    if isinstance(x, list):
        x = [to_8bit(y) for y in x]
        x = np.concatenate(x, 1)
        x = Image.fromarray(x)
    else:
        x = x - x.min()
        x = Image.fromarray(to_8bit(x))
    if show:
        x.show()
    if save:
        x.save(save)


def separate_subjects_n_slices(img_list):
    temp = [x.split('.')[0].split('_') for x in img_list]
    subject = dict()
    for y in temp:
        if int(y[0]) not in subject.keys():
            subject[int(y[0])] = []
        subject[int(y[0])] = subject[int(y[0])] + [int(y[1])]
    for k in list(subject.keys()):
        subject[k].sort()
    return subject


class DatasetFromFolderSubjects(data.Dataset):
    def __init__(self, image_dir, opt, mode):
        super(DatasetFromFolderSubjects, self).__init__()
        self.single_data = DatasetFromFolder(image_dir, opt, mode)
        self.image = self.single_data.image
        self.subject = separate_subjects_n_slices(self.image)
        self.subject_keys = list(self.subject)
        self.subject_keys.sort()

    def __len__(self):
        return len(self.subject_keys)

    def __getitem__(self, index):
        sub = self.subject_keys[index]
        image_list = self.subject[sub]

        a_list = []
        b_list = []
        for image in image_list:
            image_name = str(sub) + '_' + str(image) + '.png'
            a, b = self.single_data.__getitem__(self.image.index(image_name))
            a_list.append(a.unsqueeze(0))
            b_list.append(b.unsqueeze(0))

        a_list = torch.cat(a_list, 0)
        b_list = torch.cat(b_list, 0)

        return a_list, b_list


import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def get_transforms(input_size=256, resize=286, need=('train', 'test')):
    transformations = {}
    if 'train' in need:
        transformations['train'] = A.Compose([
            A.Resize(resize, resize),
            A.RandomCrop(height=input_size, width=input_size, p=1.),
            #A.ShiftScaleRotate(p=0.7),
            #A.HorizontalFlip(p=0.5),
            #A.VerticalFlip(p=0.5),
            #A.RandomBrightnessContrast(p=0.5),
            #A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0, additional_targets={'target': 'image'})
    if 'test' in need:
        transformations['test'] = A.Compose([
            A.Resize(resize, resize),
            A.CenterCrop(height=input_size, width=input_size, p=1.),
            #A.Normalize(p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0, additional_targets={'target': 'image'})
    return transformations


class DatasetFromFolder(data.Dataset):
    def __init__(self, root, path, opt, mode, unpaired=False):
        super(DatasetFromFolder, self).__init__()
        self.opt = opt
        self.mode = mode
        self.unpaired = unpaired

        self.a_path, self.b_path = (os.path.join(root, x) for x in path.split('_'))

        self.image_a = sorted([x.split('/')[-1] for x in glob.glob(self.a_path + '/*')])
        if self.unpaired:
            self.image_b = [x.split('/')[-1] for x in glob.glob(self.b_path + '/*')]
            random.shuffle(self.image_b)
        else:
            self.image_b = self.image_a

        self.resize = opt.resize
        self.transforms = get_transforms(resize=self.resize)[mode]

    def __len__(self):
        return len(self.image_a)

    def __getitem__(self, index):
        a = Image.open(join(self.a_path, self.image_a[index]))  #.convert('RGB') (DESS: 294>286) (PAIN: 224>286)
        b = Image.open(join(self.b_path, self.image_b[index]))
        a = np.array(a).astype(np.float32)
        b = np.array(b).astype(np.float32)
        a = a / a.max()
        b = b / b.max()

        augmented = self.transforms(image=a, target=b)
        a, b = augmented['image'], augmented['target']
        a = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(a)
        b = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(b)
        return a, b


if __name__ == '__main__':
    from dotenv import load_dotenv
    import argparse
    load_dotenv('.env')

    parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
    # Data
    parser.add_argument('--dataset', type=str, default='FlySide')
    parser.add_argument('--bysubject', action='store_true', dest='bysubject', default=False)
    parser.add_argument('--direction', type=str, default='weakxy_orixy', help='a2b or b2a')
    parser.add_argument('--flip', action='store_true', dest='flip', default=False, help='image flip left right')
    parser.add_argument('--resize', type=int, default=0)
    parser.add_argument('--mode', type=str, default='dummy')
    parser.add_argument('--port', type=str, default='dummy')
    opt = parser.parse_args()
    opt.bysubject = False
    opt.resize = 286
    #  Dataset
    if opt.bysubject:
        Dataset = DatasetFromFolderSubjects
    else:
        Dataset = DatasetFromFolder

    train_set = Dataset(root=os.environ.get('DATASET') + opt.dataset + '/train/', path=opt.direction,
                        opt=opt, mode='train')

    x = train_set.__getitem__(10)