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

PATCHSIZE=64
GAP = 48

class COPDDataset(Dataset):

    def __init__(self, data_dir, train=True, patientIDs=None, minibatch=4, augment=False):
        self.data_dir = data_dir
        self.train = train
        self.minibatch = minibatch
        if not train:
            augment = False
        self.augment = augment

        if patientIDs is None:
            maxI = 20
        else:
            maxI = -1

        # Load images
        if not patientIDs:
            for r, dirs, files in os.walk(data_dir):
                dirs = sorted(dirs)
                patientIDs = dirs
                break
        patientIDs = sorted(patientIDs)
        self.patientIDs = []
        # Create a set of patches we can create
        i=0
        self.files = []
        for pid in patientIDs:
            patientpath = os.path.join(self.data_dir, pid, 'Phase-1/RAW')
            for r, dirs, files in os.walk(patientpath):
                # print(i, patientpath, "dir exists")
                # Get the standard nii.gz file
                files = list(filter(lambda x: 'INSP' in x and 'STD' in x and 'nii.gz' in x, files))
                self.files.append(os.path.join(r, files[0]))
                self.patientIDs.append(pid)
                i += 1
            # Keep a max number of patients only
            if i == maxI:
                break

        # print(len(self.files))
        # print(self.files)

        # Create patches and cumulative patches
        self.total_patches = []
        self.cumulative_patches = [0]
        for file in self.files:
            img = nib.load(file)
            shape = img.shape
            n1 = (shape[0] - PATCHSIZE)//GAP
            n2 = (shape[1] - PATCHSIZE)//GAP
            n3 = (shape[2] - PATCHSIZE)//GAP
            patches = n1*n2*n3
            self.total_patches.append((patches, n1, n2, n3))
            self.cumulative_patches.append(self.cumulative_patches[-1] + patches)
        self.cumulative_patches = self.cumulative_patches[1:]


    def __len__(self,):
        return self.cumulative_patches[-1]

    def normalize(self, img):
        M = img.max()
        m = img.min()
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
        img = nib.load(self.files[imgidx])
        img = img.get_fdata()
        img = self.normalize(img)

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
        return {
            'image': imgpatches,
            'identifiers': torch.LongTensor(identifiers),
            # 'mask' : torch.FloatTensor(mask),
        }

