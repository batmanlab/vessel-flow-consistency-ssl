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

PATCHSIZE = 64
GAP = 48

class COPDDataset(Dataset):

    def __init__(self, data_dir, train=True, patientIDs=None, minibatch=4, augment=False, minval=-250, maxval=250):
        self.train = train
        self.minibatch = minibatch
        if self.train:
            data_dir = osp.join(data_dir, 'train/croppedCT')
        else:
            data_dir = osp.join(data_dir, 'test/croppedCT')
        self.data_dir = data_dir
        if not train:
            augment = False

        self.augment = augment
        self.minval = minval*1.0
        self.maxval = maxval*1.0

        self.files = []
        for r, dirs, files in os.walk(self.data_dir):
            self.files = list(map(lambda x: os.path.join(r, x), files))
            if patientIDs is not None:
                self.files = list(filter(lambda x: any([pid in x for pid in patientIDs]), self.files))

        print(len(self.files))
        print(self.files)

        # Create patches and cumulative patches
        self.total_patches = []
        self.cumulative_patches = [0]
        for file in self.files:
            img = load_image(file)
            shape = img.shape
            n1 = (shape[0] - PATCHSIZE)//GAP
            n2 = (shape[1] - PATCHSIZE)//GAP
            n3 = (shape[2] - PATCHSIZE)//GAP
            patches = n1*n2*n3
            self.total_patches.append((patches, n1, n2, n3))
            self.cumulative_patches.append(self.cumulative_patches[-1] + patches)

        self.cumulative_patches = self.cumulative_patches[1:]
        self.buf = dict()

    def __len__(self,):
        return self.cumulative_patches[-1]

    def normalize(self, img):
        M = self.maxval
        m = self.minval
        return (img - m)/(M-m)

    def get_patch(self, img, n1, n2, n3):
        N1 = GAP*n1
        N2 = GAP*n2
        N3 = GAP*n3
        return img[N1:N1+PATCHSIZE, N2:N2+PATCHSIZE, N3:N3+PATCHSIZE]

    def __getitem__(self, idx):
        imgidx = None
        tup = 0
        for i, cu_patches in enumerate(self.cumulative_patches):
            if idx < cu_patches:
                imgidx = i
                break
            else:
                tup = cu_patches

        _, N1, N2, N3 = self.total_patches[imgidx]
        patchnum = (idx - tup)
        n1 = patchnum%N1
        n2 = (patchnum//N1)%N2
        n3 = (patchnum//N1//N2)

        # Load image
        imgpatches = []
        identifiers = []
        if self.buf.get(imgidx) is None:
            img = load_image(self.files[imgidx])
            self.buf = dict()
            self.buf[imgidx] = img + 0
        else:
            img = self.buf[imgidx]

        img = self.normalize(img)
        #print(img.min(), img.max())

        # Get patches from the image with identifiers
        imgpatches.append(self.get_patch(img, n1, n2, n3))
        identifiers.append((imgidx, n1, n2, n3))
        for i in range(self.minibatch - 1):
            n1 = np.random.randint(N1)
            n2 = np.random.randint(N2)
            n3 = np.random.randint(N3)
            imgpatches.append(self.get_patch(img, n1, n2, n3))
            identifiers.append((imgidx, n1, n2, n3))

        imgpatches = torch.FloatTensor(imgpatches)[:, None]   # [B, 1, H, W, D]
        if imgpatches.shape[0] == 1:
            imgpatches = imgpatches.squeeze(0)
        return {
            'image': imgpatches,
            'identifiers': torch.LongTensor(identifiers),
            # 'mask' : torch.FloatTensor(mask),
        }


if __name__ == '__main__':
    ds = COPDDataset('/pghbio/dbmi/batmanlab/rohit33/COPD/', train=False)
    print(len(ds))
    print(ds[0]['image'].shape)
    print(ds[0]['image'].mean())
