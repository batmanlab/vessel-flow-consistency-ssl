import os
import numpy as np
from os import path as osp
import torch
import cv2
from torch.utils.data import Dataset
from PIL import Image
from skimage.morphology import disk, white_tophat, erosion, dilation
from scipy import ndimage as nd


def bezier_apply_random(img):
    # Apply the bezier transformation after sampling points randomly
    p0 = [0, 0]
    p3 = [1, 1]
    p1 = list(np.random.rand(2,))
    p2 = list(np.random.rand(2,))
    # Given image, solve newton raphson
    t = img*0 + 0.5
    err = 1
    while err >= 1e-4:
        f = t**3 + 3*t**2*(1-t)*p2[0] + 3*t*(1-t)**2 * p1[0] - img
        fgrad = 3*t**2 + 6*t*(1-t)*p2[0] - 3*t**2*p2[0] + 3*(1 - t)**2*p1[0] - 6*t*(1-t)*p1[0]
        t1 = t - f/fgrad
        err = np.mean(np.abs(t1 - t))
        t = t1 + 0

    bezierfunc = t**3 + 3*t**2*(1-t)*p2[1] + 3*t*(1-t)**2*p1[1]
    return bezierfunc


def patchshuffle(img, n=1000):
    H, W = img.shape[-2:]
    for i in range(n):
        patchsize = np.random.randint(4)*5 + 5
        y = np.random.randint(H - patchsize)
        x = np.random.randint(W - patchsize)
        # Take patch and shuffle it
        patch = img[:, y:y+patchsize, x:x+patchsize] + 0
        patch = patch.reshape(-1)
        np.random.shuffle(patch)
        img[:, y:y+patchsize, x:x+patchsize] = patch.reshape(-1, patchsize, patchsize)
    return img


def inpaint(img, n=20):
    H, W = img.shape[-2:]
    for i in range(n):
        patchsize = np.random.randint(50)
        y = np.random.randint(H - patchsize)
        x = np.random.randint(W - patchsize)
        # Take patch and inpaint it
        img[..., y:y+patchsize, x:x+patchsize] = np.random.rand()
    return img


PATCHSIZE = 64
def crop(ds, idx):
    # If index is not none, just use the indexed ones
    if idx is not None:
        if type(idx) == int:
            idx = [idx]
        print("Preserving indices: ", idx)
        ds.images = [ds.images[x] for x in idx]
        ds.masks  = [ds.masks[x] for x in idx]
        ds.seg    = [ds.seg[x] for x in idx]


class DriveGenesisDataset(Dataset):

    def __init__(self, data_dir, train=True, toy=False, preprocessing=False, augment=True, idx=None):
        self.data_dir = data_dir
        self.train = train
        self.disk = disk(6)
        self.preprocessing = preprocessing
        self.toy = toy
        self.trainstr = 'training' if train else 'test'
        if not train:
            augment = False
        self.augment = augment
        # Load images
        for r, dirs, images in os.walk(osp.join(self.data_dir, self.trainstr, 'images')):
            self.images = map(lambda x: osp.join(r, x), filter(lambda x: x.endswith('tif'), images))
            self.images = sorted(list(self.images))
            break

        for r, dirs, images in os.walk(osp.join(self.data_dir, self.trainstr, 'mask')):
            self.masks = map(lambda x: osp.join(r, x), filter(lambda x: x.endswith('gif'), images))
            self.masks = sorted(list(self.masks))
            break

        for r, dirs, images in os.walk(osp.join(self.data_dir, self.trainstr, '1st_manual')):
            self.seg = map(lambda x: osp.join(r, x), images)
            self.seg = sorted(list(self.seg))
            break

        crop(self, idx)


    def __len__(self,):
        if not self.augment:
            return len(self.images)
        return 8*len(self.images)


    def __getitem__(self, idx):
        flip = 0
        rot = 0
        if self.augment:
            flip = idx%8
            flip, rot = flip%2, flip//2
            idx = idx//8

        # Set up the mask
        mask = self.masks[idx]
        mask = Image.open(mask).resize((512, 512))
        if flip:
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.rotate(90*rot)
        mask = np.array(mask)/255.0

        if len(mask.shape) == 3:
            mask = mask[:, :, 0]

        # Erode the mask a bit more
        mask = cv2.erode(mask.astype(np.uint8), np.ones((7, 7), dtype=np.uint8), iterations=1)
        mask = mask[None]

        # Get image from data or gt segmentation
        if not self.toy:
            img = self.images[idx]
        else:
            img = self.seg[idx]

        # Open the image
        img = Image.open(img).resize((512, 512))
        if flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img = img.rotate(90*rot)
        img = np.array(img)/255.0

        if len(img.shape) == 3:
            img = img[..., 1]  # extract green channel only
            # Apply more preprocessing
            if self.preprocessing and not self.toy:
                img = (img - img.min())/(img.max() - img.min() + 1e-10)
                img = 1 - img
                img = img * mask[0]
                img = nd.gaussian_filter(img, 0.45)
                img = white_tophat(img, self.disk)
                img = img * mask[0]
                img = (img - img.min())/(img.max() - img.min() + 1e-10)
            else:
                if not self.toy:
                    img = nd.gaussian_filter(img, 0.45)
                img = (img - img.min())/(img.max() - img.min() + 1e-10)


        img = img[None]
        # Return [1, H, W] image and [1, H, W]

        genesisimg = img + 0
        # Apply bezier curve to it
        genesisimg = bezier_apply_random(genesisimg)
        genesisimg = patchshuffle(genesisimg)
        genesisimg = inpaint(genesisimg)

        return {
            'gt': torch.FloatTensor(img),
            'image': torch.FloatTensor(genesisimg),
            'mask' : torch.FloatTensor(mask),
        }



