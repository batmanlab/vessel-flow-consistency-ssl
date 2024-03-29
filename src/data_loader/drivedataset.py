import os
import numpy as np
from os import path as osp
import torch
import cv2
from torch.utils.data import Dataset
from PIL import Image
from skimage.morphology import disk, white_tophat, erosion, dilation
from scipy import ndimage as nd

def crop(ds, idx):
    # If index is not none, just use the indexed ones
    if idx is not None:
        if type(idx) == int:
            idx = [idx]
        print("Preserving indices: ", idx)
        ds.images = [ds.images[x] for x in idx]
        ds.masks  = [ds.masks[x] for x in idx]
        ds.seg    = [ds.seg[x] for x in idx]


class DriveDataset(Dataset):

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

        img = img[None]
        # Return [1, H, W] image and [1, H, W]
        return {
            'image': torch.FloatTensor(img),
            'mask' : torch.FloatTensor(mask),
        }



'''
This dataset will use CLAHE and gamma correction
'''
class DriveContrastDataset(Dataset):

    def __init__(self, data_dir, train=True, toy=False, preprocessing=False, augment=True, idx=None):
        self.data_dir = data_dir
        self.train = train
        self.disk = disk(6)
        self.preprocessing = preprocessing
        self.toy = toy
        self.trainstr = 'training' if train else 'test'
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
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
            self.seg = map(lambda x: osp.join(r, x), filter(lambda x: x.endswith('gif'), images))
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

        img = Image.open(img).resize((512, 512))
        if flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img = img.rotate(90*rot)
        img = np.array(img)/255.0
        if len(img.shape) == 3:
            img = img[..., 1]  # extract green channel only
            # Apply more preprocessing
            if self.preprocessing and not self.toy:
                img = (img - img.min())/(img.max() - img.min())
                img = (img)*mask[0]
                img = nd.gaussian_filter(img, 0.45)
                img = img*255
                img = img.astype(np.uint8)
                img = self.clahe.apply(img)   # Apply CLAHE
                img = img/255.0
                img = (img - img.min())/(img.max() - img.min())
                img = np.abs(img)**(1.7)
                img = cv2.medianBlur(img.astype(np.float32), 3)
                img = (img - img.min())/(img.max() - img.min())
            else:
                if not self.toy:
                    img = nd.gaussian_filter(img, 0.45)

        img = img[None]
        # Return [1, H, W] image and [1, H, W]
        return {
            'image': torch.FloatTensor(img),
            'mask' : torch.FloatTensor(mask),
        }
