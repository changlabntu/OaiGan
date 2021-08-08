import numpy as np
from collections import Counter
import glob, os, time
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from utils.data_utils import imagesc, to_8bit
from PIL import Image
import cv2
from dotenv import load_dotenv
load_dotenv('.env')


def load_OAI_var():
    all_path = os.path.join(os.path.expanduser('~'), 'Dropbox') + '/TheSource/OAIDataBase/OAI_Labels/'
    print(all_path)
    all_var = glob.glob(all_path + '*.npy')
    all_var.sort()
    v = dict()
    for var in all_var:
        name = var.split('/')[-1].split('.')[0]
        v[name] = np.load(var, allow_pickle=True)

    return v


def get_OAI_pain_labels():
    Labels = dict()
    v = load_OAI_var()

    FreL_uni = v['fre_pain_l'][np.searchsorted(v['ID_main'], v['ID_uni_fre_pain'])]
    FreR_uni = v['fre_pain_r'][np.searchsorted(v['ID_main'], v['ID_uni_fre_pain'])]

    quality = np.logical_and(v['fail_uni_l'] == 0, v['fail_uni_r'] == 0)

    select_condition = 'np.logical_and(quality == 1, abs(v["WOMP_uni_l"]-v["WOMP_uni_r"]) >= 3)'
    pick = eval(select_condition)

    Labels['label'] = FreR_uni[pick]
    Labels['ID_selected'] = v['ID_uni_fre_pain'][pick]

    return Labels['label']


def pain_prepare():
    mri_left = np.load('/media/ghc/GHc_data1/Dataset/OAI_uni_pain/unilateral_pain_left_womac3.npy')
    mri_right = np.load('/media/ghc/GHc_data1/Dataset/OAI_uni_pain/unilateral_pain_right_womac3.npy')
    label = get_OAI_pain_labels()

    for i in range(71 * 7):
        for s in range(0, 23, 1):
            if label[i] == 0:
                s_a = mri_left[i, 0, :, :, s]
                s_b = mri_right[i, 0, :, :, s]
            elif label[i] == 1:
                s_a = mri_right[i, 0, :, :, s]
                s_b = mri_left[i, 0, :, :, s]
            imagesc(s_a, show=False, save=os.environ.get('DATASET') + 'painfull/train/a/' + str(i) + '_' + str(s) + '.png')
            imagesc(s_b, show=False, save=os.environ.get('DATASET') + 'painfull/train/b/' + str(i) + '_' + str(s) + '.png')

    for i in range(71 * 7, 710):
        for s in range(0, 23, 1):
            if label[i] == 0:
                s_a = mri_left[i, 0, :, :, s]
                s_b = mri_right[i, 0, :, :, s]
            elif label[i] == 1:
                s_a = mri_right[i, 0, :, :, s]
                s_b = mri_left[i, 0, :, :, s]
            imagesc(s_a, show=False, save=os.environ.get('DATASET') + 'paired_images/painfull/test/a/' + str(i) + '_' + str(s) + '.png')
            imagesc(s_b, show=False, save=os.environ.get('DATASET') + 'paired_images/painfull/test/b/' + str(i) + '_' + str(s) + '.png')


def linear_registration(im1, im2, warp_mode, show=True):
    # try registration using open CV
    # use cv2.findTransformECC

    im1 = to_8bit(im1)[:, :, 0]#cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2 = to_8bit(im2)[:, :, 0]#cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    sz = im1.shape
    if warp_mode == 3:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)
    number_of_iterations = 5000
    termination_eps = 1e-10
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
    (cc, warp_matrix) = cv2.findTransformECC(im1, im2, warp_matrix, warp_mode, criteria=criteria)
    if warp_mode == 3:
        im2_aligned = cv2.warpPerspective(im2, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return im2_aligned


def make_compare(im1, im2):
    compare = np.zeros((im1.shape[0], im1.shape[1], 3))
    compare[:, :, 0] = im1
    compare[:, :, 2] = im1
    compare[:, :, 1] = im2
    return compare


def create_registered(alist, blist, destination):
    for i in range(len(blist)):
        print(i)
        im1 = np.array(Image.open(blist[i]))
        im2 = np.array(Image.open(alist[i]))
        try:
            new = linear_registration(im1=im1, im2=im2, warp_mode=0, show=False)
            imagesc(new, show=False, save=destination+alist[i].split('/')[-1])
        except:
            try:
                new = linear_registration(im1=im1, im2=im2, warp_mode=1, show=False)
                imagesc(new, show=False, save=destination + alist[i].split('/')[-1])
            except:
                imagesc(im2, show=False, save=destination + alist[i].split('/')[-1])


#create_registered(alist=sorted(glob.glob('/home/ghc/Dropbox/Z_DL/scripts/lightning_pix2pix/dataset/pain/train/a/*')),
#                  blist=sorted(glob.glob('/home/ghc/Dropbox/Z_DL/scripts/lightning_pix2pix/dataset/pain/train/b/*')),
#                  destination='/home/ghc/Dropbox/Z_DL/scripts/lightning_pix2pix/dataset/pain1/train/a/')

if __name__ == '__main__':
    #pain_prepare()

    source = os.environ.get('DATASET') + 'painfull/'
    create_registered(alist=sorted(glob.glob(source + '/test/a/*')),
                      blist=sorted(glob.glob(source + '/test/b/*')),
                      destination=source + '/test/aregis/')

#imagesc(make_compare(im1[:, :, 0], im2[:, :, 0]))
#imagesc(make_compare(im1[:, :, 0], nb))