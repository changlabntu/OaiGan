import numpy as np
import tifffile as tiff
import os, glob
from scipy import stats
import pandas as pd
from utils.data_utils import imagesc
from PIL import Image

aa = sorted(glob.glob('/media/ExtHDD01/logs/segA0/*'))
x = []
for a in aa:
    x.append(np.expand_dims(np.array(tiff.imread(a))[:, :], 2))
x = np.concatenate(x, 2)
s = stats.tstd(x, axis=2)
m = np.mean(x, 2)
d = np.divide(m, s)
d[np.isnan(d)] = 0
np.save('d1.npy', d)

# pain score
df = pd.read_csv('/media/ExtHDD01/OAI/OAI_extracted/OAI00womac3/OAI00womac3.csv')
labels = [(x,) for x in df.loc[df['SIDE'] == 1, 'P01KPN#EV'].astype(np.int8)]

PR = df.loc[df['SIDE'] == 1, ['V00WOMKP#']].values[497:, 0]
PL = df.loc[df['SIDE'] == 2, ['V00WOMKP#']].values[497:, 0]
PD = np.abs(PR-PL)

# bml image
#source = '/media/ExtHDD01/Dataset/paired_images/womac3/test/'
source = 'outputs/results/'
alist = sorted(glob.glob(source + 'seg1/*'))
blist = sorted(glob.glob(source + 'seg1b/*'))

abml = []
a3d = []
for a in alist:
    a = tiff.imread(a)
    #a[a < 0.5] = 0
    a = (a >= 0.5)
    abml.append(a.sum())
    a3d.append(np.expand_dims(a, 2))

bbml = []
b3d = []
for b in blist:
    b = tiff.imread(b)
    #b[b < 0] = 0
    b = (b >= 0.5)
    bbml.append(b.sum())
    b3d.append(np.expand_dims(b, 2))

abml = np.array(abml)
bbml = np.array(bbml)

bbml = np.reshape(bbml, (4899//23, 23)).sum(1)
abml = np.reshape(abml, (4899//23, 23)).sum(1)

print(stats.ttest_rel(abml, bbml))

if 1:
    a33d = np.concatenate(a3d, 2)
    b33d = np.concatenate(b3d, 2)
    a33d = np.reshape(a33d, (384, 384, 4899//23, 23))
    b33d = np.reshape(b33d, (384, 384, 4899//23, 23))

    np.save('abml', abml)
    np.save('bbml', bbml)
    np.save('PR', PR)
    np.save('PL', PL)
    np.save('PD', PD)

