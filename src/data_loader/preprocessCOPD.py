import os
import numpy as np
from os import path as osp
import torch
import cv2
from torch.utils.data import Dataset
from PIL import Image
from skimage.morphology import disk, white_tophat, erosion, dilation
from scipy import ndimage as nd
import SimpleITK as sitk
from scipy.ndimage.morphology import binary_erosion
import nibabel as nib
import argparse

def getminmax(x, H):
    minx = np.min(x)
    minx = max(0, minx-16)
    maxx = np.max(x) + 1
    maxx = min(H, maxx+16)
    return minx, maxx

parser = argparse.ArgumentParser()
parser.add_argument('--patientID', required=True, type=str, nargs='+')
parser.add_argument('--minval', type=float, default=-900)
parser.add_argument('--maxval', type=float, default=250)
parser.add_argument('--train', type=int, default=1)

# `main` part of code
args = parser.parse_args()
# Get required args
patients = args.patientID
print(patients)

for patientID in patients:
    # Get dirpath to get lobe file and CT scan
    dirpath = "/pghbio/dbmi/batmanlab/Data/COPDGene/Images/{}/Phase-1/RAW/".format(patientID)
    for r, dirs, files in os.walk(dirpath):
        files = list(filter(lambda x: 'INSP' in x and 'STD' in x and 'nii.gz' in x, files))
        ctscanfile = os.path.join(r, files[0])
        break

    # Load lobe
    ### Load Lobes
    dirpath = "/pghbio/dbmi/batmanlab/Data/COPDGene/Images/{}/Phase-1/LobeSegmentation/".format(patientID)
    lobefiles = []
    for r, dirs, files in os.walk(dirpath):
        files = list(map(lambda x: os.path.join(r, x), files))
        files = list(filter(lambda x: 'INSP' in x and 'STD' in x and 'Lobes' in x and '/.' not in x, files))
        lobefiles.extend(files)

    # Get metadata file to load it
    lobefile = lobefiles[0] if 'mhd' in lobefiles[0] else lobefiles[1]
    print(ctscanfile)
    print(lobefile)

    # Load image and label
    img = sitk.ReadImage(ctscanfile)
    img = sitk.GetArrayFromImage(img)

    Lab = sitk.ReadImage(lobefile)
    lab = sitk.GetArrayFromImage(Lab)[:, ::-1] > 0   # Make binary label
    lab = binary_erosion(lab, np.ones((3,3,3)))
    assert img.shape == lab.shape

    ## Original image, save a copy
    origimg = img + 0

    # Threshold appropriately
    img[img < args.minval] = args.minval
    img[img > args.maxval] = args.maxval

    # For non-lung points, set to minval
    img[~lab] = args.minval

    # Crop the image
    x, y, z = np.where(lab)
    H, W, D = lab.shape
    xmin, xmax = getminmax(x, H)
    ymin, ymax = getminmax(y, W)
    zmin, zmax = getminmax(z, D)

    # Crop lab image too
    croplab = sitk.GetArrayFromImage(Lab)[:, ::-1]
    croplab = croplab[xmin:xmax, ymin:ymax, zmin:zmax] + 0

    # Crop image
    cropimg = img[xmin:xmax, ymin:ymax, zmin:zmax]
    # Pad the image to multiple such that dim = 48n + 64
    H, W, D = cropimg.shape
    # Get output
    Ho = H + (48 - (H-64)%48)%48
    Wo = W + (48 - (W-64)%48)%48
    Do = D + (48 - (D-64)%48)%48
    # Save one output for visualization
    croppedOrigImg = origimg[xmin:xmax, ymin:ymax, zmin:zmax]
    outvizimg = np.zeros((Ho, Wo, Do)) + croppedOrigImg.min()
    outvizimg[:H, :W, :D] = croppedOrigImg + 0
    # Save output for label also
    outlab = np.zeros((Ho, Wo, Do))
    outlab[:H, :W, :D] = croplab + 0

    # Save output
    outimg = np.zeros((Ho, Wo, Do)) + args.minval
    outimg[:H, :W, :D] = cropimg
    print(img.shape, cropimg.shape, outimg.shape)

    # Save it
    trainstr = 'train' if args.train else 'test'

    outfilepath = "/pghbio/dbmi/batmanlab/rohit33/COPD/{}/croppedCT/{}.nii.gz".format(trainstr, patientID)
    sitk.WriteImage(sitk.GetImageFromArray(outimg), outfilepath)

    # Save label image
    outfilepath = "/pghbio/dbmi/batmanlab/rohit33/COPD/{}/lobes/lung_{}.nii.gz".format(trainstr, patientID)
    sitk.WriteImage(sitk.GetImageFromArray(outlab), outfilepath)

    # Save cropped image
    outfilepath = "/pghbio/dbmi/batmanlab/rohit33/COPD/{}/origCT/{}_orig.nii.gz".format(trainstr, patientID)
    sitk.WriteImage(sitk.GetImageFromArray(outvizimg), outfilepath)
