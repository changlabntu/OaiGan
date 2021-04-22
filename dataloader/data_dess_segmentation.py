import numpy as np
import glob, os
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torch
import torch.utils.data as data


def resize_and_crop(pilimg, scale):

    dx = 32

    w0 = pilimg.size[0]//dx * dx
    h0 = pilimg.size[1]//dx * dx
    pilimg = pilimg.crop((0, 0, w0, h0))

    w = pilimg.size[0]
    h = pilimg.size[1]
    newW = int(w * scale)
    newH = int(h * scale)

    img = pilimg.resize((newW, newH))

    return img


class DatasetDessSegmentation(data.Dataset):
    def __init__(self, mode, direction):
        super(DatasetDessSegmentation, self).__init__()
        self.dir = '/home/ghc/Dataset/OAI_DESS_segmentation/bone_resize_B_crop_00'
        self.dir_img = self.dir + '/original'
        self.dir_masks = self.dir + '/train_masks'

        ids = [x.split('/')[-1].split('.')[0] for x in glob.glob(self.dir_masks + '/femur' + '/*')]
        if mode == 'train':
            self.ids = [x for x in ids if int(x.split('_')[0]) in list(range(10, 71))]
        if mode == 'val':
            self.ids = [x for x in ids if int(x.split('_')[0]) in list(range(1, 10)) + list(range(71, 89))]

        self.mask_list = [['1'], ['2', '3']]
        self.direction = direction

    def __getitem__(self, index):
        id = self.ids[index]

        #img_size = (256, 256)
        img_size = (240, 272)

        img = Image.open(os.path.join(self.dir_img, id + '.png'))
        #img = np.array(img.resize(img_size))
        img = resize_and_crop(img, 0.5)
        img = np.array(img)
        img = img/img.max()

        # load the first mask to get the size
        if 0:
            m_all = []
            for ml in self.mask_list:
                m_one = []
                for m in ml:
                    x = Image.open(os.path.join(self.dir_masks, m, id + '.png'))
                    x = resize_and_crop(x, 0.5)
                    m_one.append(np.array(x))

                m_all.append(np.sum(m_one, 0))
            mask = np.concatenate([np.expand_dims(x, 0) for x in m_all], 0)

        if 1:
            mask = np.zeros((272, 240))
            for i, ml in enumerate(self.mask_list):
                m_one = []
                for m in ml:
                    x = Image.open(os.path.join(self.dir_masks, m, id + '.png'))
                    x = resize_and_crop(x, 0.5)
                    m_one.append(np.array(x))
                m_one = np.sum(m_one, 0)
                mask[m_one > 0] = i + 1

        a = torch.from_numpy(img).unsqueeze(0).type(torch.float32)
        b = torch.from_numpy(mask).type(torch.float32)

        a = torch.cat([a] * 3, 0)

        if self.direction == "a2b":
            return id, a, b
        elif self.direction == "b2a":
            return id, b, a

    def __len__(self):
        return len(self.ids)


if __name__ == "__main__":
    train_set = DatasetDessSegmentation(mode='val', direction='a2b')
    x = train_set.__getitem__(50)

    for i in range(len(train_set)):
        x = train_set.__getitem__(i)


