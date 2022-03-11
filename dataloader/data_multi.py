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
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import tifffile as tiff


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


def get_transforms(crop_size, resize, additional_targets, need=('train', 'test')):
    transformations = {}
    if 'train' in need:
        transformations['train'] = A.Compose([
            A.Resize(resize, resize),
            A.RandomCrop(height=crop_size, width=crop_size, p=1.),
            ToTensorV2(p=1.0),
        ], p=1.0, additional_targets=additional_targets)#{'1': '0'})
    if 'test' in need:
        transformations['test'] = A.Compose([
            A.Resize(resize, resize),
            A.CenterCrop(height=crop_size, width=crop_size, p=1.),
            ToTensorV2(p=1.0),
        ], p=1.0, additional_targets=additional_targets)#{'1': '0'})
    return transformations


class MultiData(data.Dataset):
    """
    Multiple unpaired data ccombined
    """
    def __init__(self, root, path, opt, mode, transforms=None, filenames=False):
        super(MultiData, self).__init__()
        self.opt = opt
        self.mode = mode
        self.filenames = filenames
        paired_path = path.split('%')
        self.subset = []
        for p in range(len(paired_path)):
            self.subset.append(PairedData(root=root, path=paired_path[p], opt=opt, mode=mode, filenames=filenames, bysubject=self.opt.bysubject))

    def __len__(self):
        return min([len(x) for x in self.subset])

    def __getitem__(self, index):
        outputs_all = ()
        filenames_all = ()
        if self.filenames:
            for i in range(len(self.subset)):
                outputs, filenames = self.subset[i].__getitem__(index)
                outputs_all = outputs_all + outputs
                filenames_all = filenames_all + filenames
            return outputs_all, filenames_all
        else:
            for i in range(len(self.subset)):
                outputs = self.subset[i].__getitem__(index)
                outputs_all = outputs_all + outputs
            return outputs_all


class PairedData(data.Dataset):
    """
    Paired images with the same file name from different folders
    """
    def __init__(self, root, path, opt, mode, transforms=None, filenames=False, bysubject=False):
        super(PairedData, self).__init__()
        self.opt = opt
        self.mode = mode
        self.filenames = filenames
        self.bysubject = bysubject

        self.all_path = list(os.path.join(root, x) for x in path.split('_'))

        # get name of images from the first folder
        self.images = sorted([x.split('/')[-1] for x in glob.glob(self.all_path[0] + '/*')])
        subjects = sorted(list(set([x.replace('_' + x.split('_')[-1], '') for x in self.images])))
        self.subjects = dict()
        for s in subjects:
            self.subjects[s] = sorted([x for x in self.images if x.replace('_' + x.split('_')[-1], '') == s])

        if self.opt.resize == 0:
            self.resize = np.array(Image.open(join(self.all_path[0], self.images[0]))).shape[1]
        else:
            self.resize = self.opt.resize

        if self.opt.cropsize == 0:
            self.cropsize = self.resize
        else:
            self.cropsize = self.opt.cropsize

        if transforms is None:
            additional_targets = dict()
            for i in range(1, 100):#len(self.all_path)):
                additional_targets[str(i).zfill(4)] = 'image'
            self.transforms = get_transforms(crop_size=self.cropsize,
                                             resize=self.resize,
                                             additional_targets=additional_targets)[mode]
        else:
            self.transforms = transforms

    def __len__(self):
        if not self.bysubject:
            return len(self.images)
        else:
            return len(self.subjects.keys())

    def load_img(self, path):
        x = Image.open(path)  #(DESS: 294>286) (PAIN: 224>286)
        x = np.array(x).astype(np.float32)
        if x.max() > 0:  # scale to 0-1
            x = x / x.max()
        if len(x.shape) == 2:  # if grayscale
            x = np.expand_dims(x, 2)
        x = np.concatenate([x]*3, 2)
        return x

    def __getitem__(self, index):
        filenames = ()
        inputs = dict()
        if not self.bysubject:
            for i in range(len(self.all_path)):
                name = join(self.all_path[i], self.images[index])
                filenames = filenames + (name,)
                inputs[str(i).zfill(4)] = self.load_img(name)
            inputs['image'] = inputs.pop('0000')
        else:
            subject = sorted(self.subjects.keys())[index]
            for i in range(len(self.all_path)):
                a_input = dict()
                a_filename = []

                selected = sorted(self.subjects[subject])
                for j in range(len(selected)):
                    a_slice = join(self.all_path[i], selected[j])
                    a_input[str(j).zfill(4)] = self.load_img(a_slice)
                    a_filename.append(a_slice)
                a_input['image'] = a_input.pop('0000')  # the first image in albumentation need to be name "image"
                inputs[i] = a_input
                filenames = filenames + (a_filename, )

        # Do augmentation
        outputs = ()
        if not self.bysubject:
            augmented = self.transforms(**inputs)
            augmented['0000'] = augmented.pop('image')  # 'change image back to 0'
            for k in sorted(list(augmented.keys())):
                if self.opt.n01:
                    outputs = outputs + (augmented[k],)
                else:
                    outputs = outputs + (transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(augmented[k]), )
        else:
            for i in range(len(self.all_path)):
                a_output = []
                augmented = self.transforms(**inputs[i])
                augmented['0000'] = augmented.pop('image')  # 'change image back to 0'
                for k in sorted(list(augmented.keys())):
                    if self.opt.n01:
                        a_output.append(augmented[k].unsqueeze(3))
                    else:
                        a_output.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(augmented[k]).unsqueeze(3))
                a_output = torch.cat(a_output, 3)
                outputs = outputs + (a_output, )

        # return only images or with filenames
        if self.filenames:
            return outputs, filenames
        else:
            return outputs


def save_segmentation(dataset, names, destination, use_t2d):
    """
    turn images into segmentation and save it
    """
    os.makedirs(destination, exist_ok=True)
    seg = torch.load('submodels/model_seg_ZIB_res18_256.pth').cuda()
    t2d = torch.load('submodels/tse_dess_unet32.pth')
    seg.eval()
    t2d.eval()
    for i in range(len(dataset)):
        x = dataset.__getitem__(i)[0].unsqueeze(0).cuda()
        if use_t2d:
            x = t2d(x)[0]
        out = seg(x)
        out = torch.argmax(out, 1).squeeze().detach().cpu().numpy().astype(np.uint8)
        tiff.imsave(destination + names[i], out)


if __name__ == '__main__':
    from dotenv import load_dotenv
    import argparse
    load_dotenv('.env')

    parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
    # Data
    parser.add_argument('--dataset', type=str, default='womac3')
    parser.add_argument('--bysubject', action='store_true', dest='bysubject', default=False)
    #parser.add_argument('--direction', type=str, default='xyweak_xyori_xyorisb%zyweak_zyori', help='a2b or b2a')
    #parser.add_argument('--direction', type=str, default='a_b', help='a2b or b2a')
    parser.add_argument('--direction', type=str, default='a_b', help='a2b or b2a')
    parser.add_argument('--flip', action='store_true', dest='flip', default=False, help='image flip left right')
    parser.add_argument('--resize', type=int, default=0)
    parser.add_argument('--cropsize', type=int, default=0)
    parser.add_argument('--n01', action='store_true', dest='n01', default=False)
    parser.add_argument('--gray', action='store_true', dest='gray', default=False, help='only use 1 channel')
    parser.add_argument('--mode', type=str, default='dummy')
    parser.add_argument('--port', type=str, default='dummy')
    opt = parser.parse_args()

    root = os.environ.get('DATASET') + opt.dataset + '/test/'
    source = 'a_b'
    destination = 'bseg/'
    opt.direction = source
    #opt.bysubject = True
    opt.cropsize = 384
    dataset = MultiData(root=root, path=opt.direction, opt=opt, mode='train', filenames=True)
    x = dataset.__getitem__(14)
    opt.bysubject = True
    dataset = MultiData(root=root, path=opt.direction, opt=opt, mode='train', filenames=True)
    x3d = dataset.__getitem__(0)

    #  Dataset
    if 0:
        #x = train_set.__getitem__(100)
        names = [x.split('/')[-1] for x in sorted(glob.glob(root + source + '/*'))]
        save_segmentation(dataset=dataset,
                          names=names,
                          destination=root + destination, use_t2d=True)

    if 0:
        root = os.environ.get('DATASET') + opt.dataset + '/test/'
        source = 'aregis1/'
        mask = 'aseg/'
        destination = 'amask/'
        images = [x.split('/')[-1] for x in sorted(glob.glob(root + source + '*'))]

        os.makedirs(root + destination, exist_ok=True)
        for im in images:
            x = np.array(Image.open(root + source + im))
            m = np.array(Image.open(root + mask + im))
            m = (m == 1) + (m == 3)
            masked = np.multiply(x, m)
            tiff.imsave(root + destination + im, masked)