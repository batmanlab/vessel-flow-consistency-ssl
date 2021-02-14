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


class RITEDataset(Dataset):

    def __init__(self, data_dir, train=True, augment=True):
        self.data_dir = data_dir
        self.train = train
        self.augment = augment

        trainstr = 'training' if train else 'test'
        self.data_dir = osp.join(self.data_dir, trainstr)

        self.images = get_all_files(osp.join(self.data_dir, 'images'))
        self.segs = get_all_files(osp.join(self.data_dir, 'vessel'))


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
        seg = self.segs[idx]
        seg = Image.open(seg).resize((512, 512))
        # Flip
        if flip:
            seg = seg.transpose(Image.FLIP_LEFT_RIGHT)

        # Rotate
        seg = seg.rotate(90*rot)
        seg = np.array(seg)/255.0

        if len(seg.shape) == 3:
            seg = seg[:, :, 0]

        seg = seg[None]

        # Get image from data or gt segmentation
        img = self.images[idx]
        img = Image.open(img).resize((512, 512))
        if flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img = img.rotate(90*rot)
        img = np.array(img)/255.0

        if len(img.shape) == 3:
            img = img[..., 1]  # extract green channel only
            # Apply more preprocessing

        img = img[None]
        mask = (img > 20/255.0).astype(int)
        # Return [1, H, W] image and [1, H, W]
        return {
            'image': torch.FloatTensor(img),
            'mask': torch.FloatTensor(mask),
            'seg': torch.FloatTensor(seg),
        }


if __name__ == "__main__":
    ds = RITEDataset("/ocean/projects/asc170022p/rohit33/RITE_dataset", train=True)
    print(ds.images)
    print(len(ds))
    d = ds[0]
    for k, v in d.items():
        print(k, v.shape)

