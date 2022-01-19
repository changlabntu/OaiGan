import os, glob
from skimage import io
from utils.data_utils import imagesc as show


def to_patches(images, destination, zrange, dx=256, criteria=0.2):
    Z, H, W = images[list(images.keys())[0]].shape
    print((Z, H, W))

    keys = list(images.keys())
    os.makedirs(destination, exist_ok=True)
    for name in keys:
        os.makedirs(destination + name + '/', exist_ok=True)

    for s in zrange:
        for i in range(H // dx):
            for j in range(W // dx):
                name = keys[0]
                patch = images[name][s, i * dx:(i + 1) * dx, j * dx:(j + 1) * dx]
                avg = (patch > 0).mean()
                if avg >= criteria:
                    show(patch, show=False, save=destination + name + '/' + str(i) + '_' + str(j) + '_' + str(s)
                                                 + '_' + "{:.2f}".format(avg) + '.png')
                    for k in range(1, len(keys)):
                        name = keys[k]
                        patch = images[name][s, i * dx:(i + 1) * dx, j * dx:(j + 1) * dx]
                        show(patch, show=False, save=destination + name + '/' + str(i) + '_' + str(j) + '_' + str(s)
                                                     + '_' + "{:.2f}".format(avg) + '.png')


root = '/media/ExtHDD01/Dataset/paired_images/FlySide/'
"""
images: dictionary consisted of images that will be broken into patches in the same time.
Criteria will be applied based on the first image
"""
images = dict([(x, io.imread(root + x + '.tif')) for x in ['weakzy', 'votezy']])
Z = images[list(images.keys())[0]].shape[0]
to_patches(images=images, zrange=range(0, Z // 4 * 3), destination=root + 'train/', dx=256, criteria=0.2)
to_patches(images=images, zrange=range(Z // 4 * 3, Z), destination=root + 'test/', dx=256, criteria=0.2)

#images = dict([(x, io.imread(root + x + '.tif')) for x in ['weakzy', 'orizy']])
#to_patches(images, root, dx=256, criteria=0.2)