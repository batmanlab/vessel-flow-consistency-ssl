import os
import numpy as np
from os import path as osp
import torch
from torch.utils.data import Dataset
from PIL import Image

class DriveDataset(Dataset):

    def __init__(self, data_dir, train=True, toy=False):
        self.data_dir = data_dir
        self.train = train
        self.toy = toy
        self.trainstr = 'training' if train else 'test'
        # Load images
        for r, dirs, images in os.walk(osp.join(self.data_dir, self.trainstr, 'images')):
            self.images = map(lambda x: osp.join(r, x), filter(lambda x: x.endswith('tif'), images))
            self.images = sorted(list(self.images))
            break

        for r, dirs, images in os.walk(osp.join(self.data_dir, self.trainstr, '1st_manual')):
            self.masks = map(lambda x: osp.join(r, x), filter(lambda x: x.endswith('gif'), images))
            self.masks = sorted(list(self.masks))
            break

    def __len__(self,):
        if not self.train:
            return len(self.images)
        return 2*len(self.images)

    def __getitem__(self, idx):
        flip = 0
        if self.train:
            flip = idx%2
            idx = idx//2

        mask = self.masks[idx]
        mask = Image.open(mask)
        if flip:
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        mask = np.array(mask)/255.0
        mask = 2*mask - 1
        mask = mask[None]

        if not self.toy:
            img = self.images[idx]
            img = Image.open(img)
            if flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img = np.array(img)/255.0
            img = 2*img - 1
            img = img.transpose(2, 0, 1)
        else:
            img = mask + 0
        # Return [C, H, W] image and [1, H, W]
        #print(img.min(), img.max(), mask.min(), mask.max())
        return {
            'image': torch.FloatTensor(img),
            'mask' : torch.FloatTensor(mask),
        }
