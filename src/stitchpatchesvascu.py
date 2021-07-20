import pickle as pkl
import os
from os import path as osp
from tqdm import tqdm
import torch
import numpy as np
import argparse
import nibabel as nib
import SimpleITK as sitk
from data_loader.vascu import VascuDataset

parser = argparse.ArgumentParser()
parser.add_argument('--sigma', type=float, default=0)
parser.add_argument('--threshold', type=float, default=None)

def main():
    args = parser.parse_args()
    sigma = args.sigma
    # Get dataset and vesselness
    ds = VascuDataset('/pghbio/dbmi/batmanlab/rohit33/VascuSynth', train=False, sigma=sigma)
    filename = "/pghbio/dbmi/batmanlab/rohit33/vascutest/test_vesselness_3d.pkl"
    with open(filename, 'rb') as fi:
        data = pkl.load(fi)
    print(len(ds))
    print(len(data))
    assert  len(ds)%8 == 0
    assert  len(ds) == len(data)

    # Get vesselness
    # Vesselness and count
    # Vesselness
    for i in tqdm(range(len(ds))):
        # Init all variables
        if i%8 == 0:
            fVesselness = np.zeros((128, 128, 128))
            fImg = np.zeros((128, 128, 128))
            fGt = np.zeros((128, 128, 128))

        inp = ds[i]['image']
        gt = ds[i]['gt']
        idx = i%8
        h = (idx%2)
        w = (idx//2)%2
        d = (idx//2)//2

        fVesselness[64*h:64*(h+1), 64*w:64*(w+1), 64*d:64*(d+1)] = data[i].squeeze()
        fImg[64*h:64*(h+1), 64*w:64*(w+1), 64*d:64*(d+1)] = inp.squeeze()
        fGt[64*h:64*(h+1), 64*w:64*(w+1), 64*d:64*(d+1)] = gt.squeeze()
        # Save them
        if i%8 == 7:
            # Threshold it if possible
            if args.threshold is not None:
                fVesselness = (fVesselness >= args.threshold).astype(float)
            filename = ds.files[i//8]
            filename = osp.join("/pghbio/dbmi/batmanlab/rohit33/vascutest", "-".join(filename.split("/")[-3:-1]))
            fImg = nib.Nifti1Image(fImg, affine=np.eye(4))
            fVesselness = nib.Nifti1Image(fVesselness, affine=np.eye(4))
            fGt = nib.Nifti1Image(fGt, affine=np.eye(4))
            # Save
            # print("Saving to {}".format(filename))
            nib.save(fImg, filename + '-img.nii.gz')
            nib.save(fGt, filename + '-gt.nii.gz')
            nib.save(fVesselness, filename + '-v.nii.gz')


if __name__ == '__main__':
    main()
