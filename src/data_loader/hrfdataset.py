import os
import numpy as np
from os import path as osp
import torch
import cv2
from torch.utils.data import Dataset
from PIL import Image
from skimage.morphology import disk, white_tophat, erosion, dilation
from scipy import ndimage as nd

def get_all_files(dirname):
    # Get all files from a directory name (assuming no other directories inside)
    for r, dirs, files in os.walk(dirname):
        if dirs != []:
            assert False, 'Dircetory found inside path {}'.format(r)
        files = map(lambda x: osp.join(r, x), files)
        files = list(files)
    files = sorted(files)
    return files


class HRFDataset(Dataset):

    def __init__(self, data_dir, train=True, augment=True):
        self.data_dir = data_dir
        self.train = train
        self.augment = augment

        # Get all images, masks, and segmentations
        self.images = get_all_files(osp.join(data_dir, 'images'))
        self.masks = get_all_files(osp.join(data_dir, 'mask'))
        self.segs = get_all_files(osp.join(data_dir, 'manual1'))

        #for i, m, s in zip(self.images, self.masks, self.segs):
            #print(i, m, s)
        #input()


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

        # Set up the mask and segmentation
        mask = self.masks[idx]
        mask = Image.open(mask)
        width, height = mask.width//4, mask.height//4
        mask = mask.resize((width, height))

        seg = self.segs[idx]
        seg = Image.open(seg)
        seg = seg.resize((width, height))

        # Flip
        if flip:
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            seg = seg.transpose(Image.FLIP_LEFT_RIGHT)

        # Rotate
        mask = mask.rotate(90*rot)
        seg = seg.rotate(90*rot)

        mask = np.array(mask)/255.0
        seg = np.array(seg)/255.0

        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        if len(seg.shape) == 3:
            seg = seg[:, :, 0]

        # Erode the mask a bit more
        mask = cv2.erode(mask.astype(np.uint8), np.ones((7, 7), dtype=np.uint8), iterations=1)
        mask = mask[None]
        seg = seg[None]

        # Get image from data or gt segmentation
        img = self.images[idx]
        img = Image.open(img)
        img = img.resize((width, height))
        if flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img = img.rotate(90*rot)
        img = np.array(img)/255.0

        if len(img.shape) == 3:
            img = img[..., 1]  # extract green channel only
            # Apply more preprocessing

        img = img[None]
        # Return [1, H, W] image and [1, H, W]
        return {
            'image': torch.FloatTensor(img),
            'mask' : torch.FloatTensor(mask),
            'seg': torch.FloatTensor(seg),
        }


if __name__ == "__main__":
    ds = HRFDataset("/ocean/projects/asc170022p/rohit33/HRF_dataset")
    d = ds[0]
    for k, v in d.items():
        print(k, v.shape)

