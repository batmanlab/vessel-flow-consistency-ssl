''' 
Test script to store Frangi resutls for the VascuSynth dataset. 
To ensure a similar data structure (patchified data) as that from the learnt method, we artifically
chop the vesselness image into patches before saving.
'''
import pickle as pkl
import os
from os import path as osp
from tqdm import tqdm
import torch
import numpy as np
from skimage.filters import frangi, sato
import argparse
import nibabel as nib
import SimpleITK as sitk
from data_loader.vascu import VascuDataset

parser = argparse.ArgumentParser()
parser.add_argument('--sigma', type=float, default=0)
parser.add_argument('--threshold', type=float, default=None)

def frangi_vesselness(img):
    ves = frangi(img, np.linspace(1, 12, 12), black_ridges=False).astype(np.float32)
    return ves

def main():
    args = parser.parse_args()
    sigma = args.sigma
    # Get dataset and vesselness
    ds = VascuDataset('/pghbio/dbmi/batmanlab/rohit33/VascuSynth', train=False, sigma=sigma)
    print(len(ds))
    assert  len(ds)%8 == 0

    # Get vesselness
    # Vesselness and count
    # Vesselness
    for i in tqdm(range(len(ds))):
        # Init all variables
        if i%8 == 0:
            fVesselness = np.zeros((128, 128, 128))

        inp = ds[i]['image'].squeeze().numpy()
        gt = ds[i]['gt']
        ves = frangi_vesselness(inp)
        idx = i%8
        h = (idx%2)
        w = (idx//2)%2
        d = (idx//2)//2

        fVesselness[64*h:64*(h+1), 64*w:64*(w+1), 64*d:64*(d+1)] = ves
        # Save them
        if i%8 == 7:
            # Threshold it if possible
            if args.threshold is not None:
                fVesselness = (fVesselness >= args.threshold).astype(float)
            filename = ds.files[i//8]
            filename = osp.join("/pghbio/dbmi/batmanlab/rohit33/vascutest", "-".join(filename.split("/")[-3:-1]))
            fVesselness = nib.Nifti1Image(fVesselness, affine=np.eye(4))
            # Save
            # print("Saving to {}".format(filename))
            nib.save(fVesselness, filename + '-frangi.nii.gz')





if __name__ == '__main__':
    main()
