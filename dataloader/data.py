from os import listdir
from os.path import join
import glob
import random

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from dataloader.utils import is_image_file
from PIL import Image
import numpy as np


def get_training_set(root_dir, direction, mode):
    train_dir = join(root_dir, "train")

    return DatasetFromFolder(train_dir, direction, mode)


def get_test_set(root_dir, direction, mode):
    test_dir = join(root_dir, "test")

    return DatasetFromFolder(test_dir, direction, mode)


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


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, direction, mode):
        super(DatasetFromFolder, self).__init__()
        self.direction = direction
        self.mode = mode
        self.a_path = join(image_dir, direction.split('_')[0])
        self.b_path = join(image_dir, direction.split('_')[1])

        self.image = sorted([x.split('/')[-1] for x in glob.glob(self.a_path+'/*')])

        #transform_list = [transforms.ToTensor(),
        #                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        #self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        a = Image.open(join(self.a_path, self.image[index]))#.convert('RGB')
        b = Image.open(join(self.b_path, self.image[index]))#.convert('RGB')
        a = a.resize((286, 286), Image.BICUBIC)  # 444 > 286
        b = b.resize((286, 286), Image.BICUBIC)
        a = transforms.ToTensor()(a)
        b = transforms.ToTensor()(b)
        a = a.type(torch.float32)
        b = b.type(torch.float32)
        a = a / a.max()
        b = b / b.max()
        if 1:  # random crop
            #a = torch.nn.functional.interpolate(a.unsqueeze(0), (286, 286), mode='bicubic', align_corners=True)[0, ::]
            #b = torch.nn.functional.interpolate(b.unsqueeze(0), (286, 286), mode='bicubic', align_corners=True)[0, ::]
            if self.mode == 'train':
                w_offset = random.randint(0, max(0, 286 - 256 - 1))
                h_offset = random.randint(0, max(0, 286 - 256 - 1))
            elif self.mode == 'test':
                w_offset = 15
                h_offset = 15
            a = a[:, h_offset:h_offset + 256, w_offset:w_offset + 256]
            b = b[:, h_offset:h_offset + 256, w_offset:w_offset + 256]

        if a.shape[0] != 3:
            a = torch.cat([a] * 3, 0)
            b = torch.cat([b] * 3, 0)
        a = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(a)
        b = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(b)

        if self.mode == 'train':  # flipping
            if random.random() < 0.5:
                idx = [i for i in range(a.size(2) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                a = a.index_select(2, idx)
                b = b.index_select(2, idx)

        return a, b

    def __len__(self):
        return len(self.image)