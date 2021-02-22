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

N = 32
M = 80
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
        self.offset = offset
        if offset > 0:
            allmra = allmra[offset:] + allmra[:offset]
            alltre = alltre[offset:] + alltre[:offset]

        self.allmra = allmra
        self.alltre = alltre

        # Split according to train or test
        self.allmra = self.allmra
        self.alltre = self.alltre

        # Calculate number of patches per dimension
        self.Np = [math.ceil((DIMS[i]-64)/48.)+1 for i in range(3)]
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
        pids = [48*_ + 64 for _ in pids]   # end coordinate
        pids = [min(DIMS[i], x) for i,x in enumerate(pids)]
        pids = [x - 64 for x in pids]

        h, w, d = pids
        return img[h:h+64, w:w+64, d:d+64], [0, h, w, d]


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
        shape = img.shape
        img = self.normalize(img + 0)
        #print(img.min(), img.max(), img.dtype)
        #print(img.min(), img.max())
        img, startcoord = self.crop(img, patchid)
        return {
                'image': torch.FloatTensor(img)[None],
                'path': self.allmra[imgid],
                'gt': 0,
                'imgid': (imgid + self.offset)%len(self.allmra),
                'startcoord': torch.LongTensor(startcoord),
                'shape': torch.LongTensor(shape),
        }



class TubeTKFullDataset(TubeTKDataset):

    def __init__(self, data_dir, train=True, offset=0):
        self.data_dir = data_dir
        allmra = []
        for r, dirs, files in os.walk(data_dir):
            files = map(lambda x: os.path.join(r,x), files)
            files = filter(lambda x: 'MRA' in x, files)
            files = filter(lambda x: x.endswith('.mha'), files)
            allmra.extend(files)

        # Get an offset
        if offset > 0:
            allmra = allmra[offset:] + allmra[:offset]

        self.allmra = sorted(allmra)

        # Split according to train or test
        if train:
            self.allmra = self.allmra[:M]
        else:
            self.allmra = self.allmra[M:]
        # Calculate number of patches per dimension
        self.Np = [math.ceil((DIMS[i]-64)/48.)+1 for i in range(3)]
        self.numPatches = np.prod(np.array(self.Np))
        print("{} patches per image.".format(self.numPatches))
        #print(self.allmra)
        self.buf = dict()



if __name__ == '__main__':
    off = 31
    ds = TubeTKDataset('/ocean/projects/asc170022p/rohit33/TubeTK', offset=off)
    #print(ds.allmra)
    for i, x in enumerate(ds.allmra):
        print(i, x)

    #ds = TubeTKFullDataset('/ocean/projects/asc170022p/rohit33/TubeTK')
    print(len(ds))
    for _ in range(260):
        d  = ds[_*200]
        print(d['imgid'], d['path'])
        #print(d.shape, d.mean(), d.min(), d.max())
        #print(d['startcoord'], d['startcoord']+64, d['shape'])
