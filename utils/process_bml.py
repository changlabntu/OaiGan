import numpy as np
import tifffile as tiff
import os, glob
from scipy import stats
import pandas as pd
from utils.data_utils import imagesc

# pain score
df = pd.read_csv('/media/ExtHDD01/OAI/OAI_extracted/OAI00womac3/OAI00womac3.csv')
labels = [(x,) for x in df.loc[df['SIDE'] == 1, 'P01KPN#EV'].astype(np.int8)]

PR = df.loc[df['SIDE'] == 1, ['V00WOMKP#']].values[497:, 0]
PL = df.loc[df['SIDE'] == 2, ['V00WOMKP#']].values[497:, 0]
PD = np.abs(PR-PL)

# bml image
source = '/media/ExtHDD01/Dataset/paired_images/womac3/test/'

alist = sorted(glob.glob(source + 'abml2/*'))
blist = sorted(glob.glob(source + 'bbml2/*'))

abml = []
a3d = []
for a in alist:
    a = tiff.imread(a)
    a[a < 0.15] = 0
    abml.append(a.sum())
    a3d.append(np.expand_dims(a, 2))

bbml = []
b3d = []
for b in blist:
    b = tiff.imread(b)
    b[b < 0.15] = 0
    bbml.append(b.sum())
    b3d.append(np.expand_dims(b, 2))

abml = np.array(abml)
bbml = np.array(bbml)

bbml = np.reshape(bbml, (4899//23, 23)).sum(1)
abml = np.reshape(abml, (4899//23, 23)).sum(1)

print(stats.ttest_rel(abml, bbml))

a3d = np.concatenate(a3d, 2)
b3d = np.concatenate(b3d, 2)
a3d = np.reshape(a3d, (384, 384, 4899//23, 23))
b3d = np.reshape(b3d, (384, 384, 4899//23, 23))

np.save('abml', abml)
np.save('bbml', bbml)
np.save('PR', PR)
np.save('PL', PL)
np.save('PD', PD)

