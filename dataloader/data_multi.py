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
        self.paths = [join(image_dir, x) for x in direction.split('_')]
        self.image = sorted([x.split('/')[-1] for x in glob.glob(self.paths[0]+'/*')])

        #transform_list = [transforms.ToTensor(),
        #                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        #self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):

        img = []
        for p in self.paths:
            a = Image.open(join(p, self.image[index]))  # .convert('RGB')
            a = a.resize((286, 286), Image.BICUBIC)  # 444 > 286
            #a = transforms.ToTensor()(a0)
            a = torch.from_numpy(np.array(a)).unsqueeze(0)
            a = a.type(torch.float32)
            a = a / a.max()
            if len(a.shape) == 4:
                a = a[:, :, :, 0]
            img.append(a)

        if self.mode == 'train':
            w_offset = random.randint(0, max(0, 286 - 256 - 1))
            h_offset = random.randint(0, max(0, 286 - 256 - 1))
        elif self.mode == 'test':
            w_offset = 15
            h_offset = 15

        for i in range(len(img)):
            img[i] = img[i][:, h_offset:h_offset + 256, w_offset:w_offset + 256]

        for i in range(2):
            if img[i].shape[0] != 3:
                img[i] = torch.cat([img[i]] * 3, 0)
            img[i] = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img[i])

        return img

    def __len__(self):
        return len(self.image)


if __name__ == '__main__':
    ds = get_training_set('/media/ghc/GHc_data1/paired_images/TSE_DESS/', direction='a_b_bseg', mode='train')