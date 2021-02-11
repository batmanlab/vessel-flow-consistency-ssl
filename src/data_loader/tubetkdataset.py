import os
import math
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


DIMS = (128, 448, 448)

class TubeTKDataset(Dataset):

    def __init__(self, data_dir, train=True, offset=0):
        self.data_dir = data_dir
        allmra = []
        alltre = []
        for r, dirs, files in os.walk(data_dir):
            files = list(map(lambda x: os.path.join(r, x), files))
            tre = list(filter(lambda x: 'Aux' in x and 'tre' in x, files))
            alltre.extend(tre)
        alltre = sorted(alltre)

        # Derive MRA images based on this
        for tre in alltre:
            dirname = "/".join(tre.split('/')[:-2] + ['MRA'])
            for r, dirs, files in os.walk(dirname):
                mra = os.path.join(r, files[0])
            allmra.append(mra)

        # Get an offset
        if offset > 0:
            allmra = allmra[offset:] + allmra[:offset]
            alltre = alltre[offset:] + alltre[:offset]

        self.allmra = allmra
        self.alltre = alltre

        # Calculate number of patches per dimension
        self.Np = [math.ceil(DIMS[i]/64.) for i in range(3)]
        self.numPatches = np.prod(np.array(self.Np))
        print("{} patches per image.".format(self.numPatches))
        self.buf = dict()


    def __len__(self,):
        return len(self.allmra)*self.numPatches


    def crop(self, img, patchid):
        pids = []
        # Get patchids
        pid = patchid
        for _ in range(3):
            pids.append(pid%self.Np[_])
            pid = pid//self.Np[_]

        #print(patchid, pids)
        pids = [64*_ for _ in pids]

        h, w, d = pids
        return img[h:h+64, w:w+64, d:d+64]


    def load_image(self, imgfile):
        if self.buf.get(imgfile) is None:
            self.buf = dict()
            img = sitk.ReadImage(imgfile)
            img = sitk.GetArrayFromImage(img)*1.0
            self.buf[imgfile] = img + 0
        else:
            img = self.buf[imgfile] + 0
        return img


    def normalize(self, img):
        M = img.max()
        m = img.min()
        return 2*(img - m)/(M - m) - 1


    def __getitem__(self, idx):
        patchid = idx%self.numPatches
        imgid = idx//self.numPatches

        # Load image
        img = self.load_image(self.allmra[imgid]) + 0
        img = self.normalize(img + 0)
        #print(img.min(), img.max(), img.dtype)
        #print(img.min(), img.max())
        img = self.crop(img, patchid)
        return {
                'image': torch.FloatTensor(img)[None],
                'gt': 0,
        }



if __name__ == '__main__':
    ds = TubeTKDataset('/ocean/projects/asc170022p/rohit33/TubeTK')
    print(len(ds))
    for _ in range(60):
        d  = ds[_]['image']
        print(d.shape, d.mean(), d.min(), d.max())
