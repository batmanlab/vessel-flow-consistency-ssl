import os
from os import path as osp
import torch
from torch.utils.data import Dataset
from PIL import Image

class DriveDataset(Dataset):

    def __init__(self, data_dir, train=True):
        self.data_dir = data_dir
        self.train = train
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
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        img = np.array(Image.open(img))/255.0
        mask = self.masks[idx]
        mask = np.array(Image.open(mask))/255.0
        # Return [C, H, W] image and [1, H, W]
        return {
            'image': torch.tensor(img.transpose(2, 0, 1)),
            'mask' : torch.tensor(mask[None]),
        }
