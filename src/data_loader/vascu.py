import os
import numpy as np
from os import path as osp
import torch
import cv2
from torch.utils.data import Dataset
from PIL import Image
from skimage.morphology import disk, white_tophat, erosion, dilation
from scipy import ndimage as nd
import nibabel as nib
import SimpleITK as sitk

def load_image(imgfile):
    img = sitk.ReadImage(imgfile)
    img = sitk.GetArrayFromImage(img)
    return img

class VascuDataset(Dataset):

    def __init__(self, data_dir, train=True, sigma=0.0):
        self.data_dir = data_dir
        self.train = train
        if train:
            groups = [1, 2, 3, 4, 5]
        else:
            groups = [6, 7, 8, 9]

        self.files = []
        for gr in groups:
            for r, dirs, files in os.walk(osp.join(data_dir, 'Group{}'.format(gr))):
                files = list(map(lambda x: osp.join(r, x), files))
                files = list(filter(lambda x: 'mhd' in x, files))
                self.files.extend(files)
        #print(self.files)
        self.sigma = sigma
        self.buf = dict()

    def __len__(self,):
        return len(self.files)*8

    def load_image(self, imgfile):
        if self.buf.get(imgfile) is None:
            self.buf = dict()
            img = sitk.ReadImage(imgfile)
            img = sitk.GetArrayFromImage(img)
            self.buf[imgfile] = img + 0
        else:
            img = self.buf[imgfile] + 0
        return img

    def normalize(self, img):
        M = 255.0
        m = 0.0
        return (img - m)/(M - m)

    def crop(self, img, n=0):
        H, W, D = img.shape
        nimg = np.zeros((128, 128, 128))
        nimg[:H, :W, :D] = img + 0
        h = n%2
        w = (n//2)%2
        d = (n//2)//2
        return nimg[64*h:64*(h+1), 64*w:64*(w+1), 64*d:64*(d+1)]

    def get_noise(self, idx, shape,):
        thres = 0 if self.train else 1000
        rng = np.random.RandomState(idx + thres)
        noise = rng.randn(*shape) * self.sigma
        return noise

    def __getitem__(self, idx):
        img = self.load_image(self.files[idx//8])
        img = self.normalize(img)
        img = self.crop(img, idx%8)
        gt = (img > 0.1).astype(int)
        # Add noise to image
        img = img + self.get_noise(idx, img.shape)
        return {
                'image': torch.FloatTensor(img)[None],
                'gt': torch.LongTensor(gt)[None],
        }


if __name__ == '__main__':
    ds = VascuDataset('/pghbio/dbmi/batmanlab/rohit33/VascuSynth')
    print(len(ds))
    print(ds[0]['image'].shape)
    print(ds[0]['image'].mean())
