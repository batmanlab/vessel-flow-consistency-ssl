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
from functools import lru_cache

N = 32
M = 80
# DIMS = (128, 448, 448)

# Number of patches for each image for easy querying
trainPatchIds = [(8.0, 11.0, 11.0), (9.0, 11.0, 11.0), (11.0, 11.0, 11.0), (9.0, 11.0, 11.0), (9.0, 11.0, 11.0), (8.0, 11.0, 11.0), (10.0, 11.0, 11.0), (9.0, 11.0, 11.0), (11.0, 11.0, 11.0), (9.0, 11.0, 11.0), (9.0, 11.0, 11.0), (9.0, 11.0, 11.0), (10.0, 11.0, 11.0), (8.0, 11.0, 11.0), (8.0, 11.0, 11.0), (10.0, 11.0, 11.0), (9.0, 11.0, 11.0), (9.0, 11.0, 11.0), (8.0, 11.0, 11.0), (9.0, 11.0, 11.0)]

trainNumPatches = [968.0, 1089.0, 1331.0, 1089.0, 1089.0, 968.0, 1210.0, 1089.0, 1331.0, 1089.0, 1089.0, 1089.0, 1210.0, 968.0, 968.0, 1210.0, 1089.0, 1089.0, 968.0, 1089.0]

# Testing
testPatchIds = [(10.0, 11.0, 11.0), (9.0, 11.0, 11.0), (9.0, 11.0, 11.0)]
testNumPatches = [1210.0, 1089.0, 1089.0]

class Vessel12Dataset(Dataset):

    def __init__(self, data_dir, train=True, offset=0, pad=0):
        self.data_dir = data_dir
        if train:
            self.patchids = trainPatchIds
            self.numpatches = trainNumPatches
        else:
            self.patchids = testPatchIds
            self.numpatches = testNumPatches

        self.pad = pad

        # Keep count of cumulative number of patches
        self.cnumpatches = np.cumsum(self.numpatches)

        trainstr = 'train' if train else 'test'
        self.dirname = osp.join(data_dir, trainstr)
        # Read images
        for r, dirs, files in os.walk(self.dirname):
            files = map(lambda x: osp.join(r, x), files)
            files = filter(lambda x: x.endswith('mhd'), files)
            files = sorted(list(files))
            break

        self.files = files

        # Get annotations files
        csvs = None
        if os.path.exists(osp.join(self.dirname, 'Annotations')):
            for r, dirs, csvs in os.walk(osp.join(self.dirname, 'Annotations')):
                csvs = map(lambda x: os.path.join(r, x), csvs)
                csvs = filter(lambda x: x.endswith('csv'), csvs)
                csvs = sorted(list(csvs))
                break
            self.csvs = csvs


    def __len__(self,):
        return int(self.cnumpatches[-1])


    def __getitem__(self, idx):
        imgid, patchid = self._get_ids(idx)
        # Load image
        img = self.load_image(self.files[imgid]) + 0
        shape = img.shape
        # Crop the image
        img, startcoord = self.crop(img, imgid, patchid)

        img = self.normalize(img + 0)

        return {
                'image': torch.FloatTensor(img)[None],
                'gt': 0,
                'startcoord': torch.LongTensor(startcoord),
                'shape': torch.LongTensor(shape),
                'imgid': imgid,
        }

    def crop(self, img, imgid, patchid):
        # Crop the relevant part of the image
        shape = img.shape
        pids = []
        totalPatches = self.patchids[imgid]
        # Get patchids
        pid = patchid
        for _ in range(3):
            pids.append(pid%totalPatches[_])
            pid = pid//totalPatches[_]

        # Convert these patch indices into coordinates to crop from
        pids = [48*_ + 64 for _ in pids]  # find last coordinate
        pids = [min(shape[i], x) for i,x in enumerate(pids)]   # crop it
        pids = [_ - 64 for _ in pids]  # remove patch size
        #pids = [min(shape[i]-64, x) for i,x in enumerate(pids)]    # Crop it if it goes outside
        pids = [int(x) for x in pids]
        h, w, d = pids

        # If pad is not zero, pad appropriately
        pad = self.pad
        if pad > 0:
            H, W, D = img.shape
            imgbig = np.zeros((H + 2*pad, W + 2*pad, D + 2*pad)) - 900
            imgbig[pad:-pad, pad:-pad, pad:-pad] = img
            return imgbig[h:h+64+2*pad, w:w+64+2*pad, d:d+64+2*pad],  [imgid, h, w, d]
        else:
            return img[h:h+64, w:w+64, d:d+64],  [imgid, h, w, d]


    def load_image(self, imgfile):
        img = sitk.ReadImage(imgfile)
        img = sitk.GetArrayFromImage(img)*1.0
        return img


    def normalize(self, img):
        M = 250
        m = -900
        img[img > M] = M
        img[img < m] = m
        return 2*(img - m)/(M - m) - 1


    def _get_ids(self, idx):
        # Given master idx, derive an image index and a patch index within the image
        imgid = 0
        while idx >= self.numpatches[imgid]:
            idx -= self.numpatches[imgid]    # Delete that many number of patches
            imgid += 1
        return imgid, idx


if __name__ == '__main__':
    ds = Vessel12Dataset('/ocean/projects/asc170022p/rohit33/VESSEL12', train=False)
    print(len(ds))
    for _ in range(300):
        d  = ds[_]['image']
        print(ds[_]['startcoord'], 64 + ds[_]['startcoord'], ds[_]['shape'])
        print(d.shape, d.mean(), d.min(), d.max())
