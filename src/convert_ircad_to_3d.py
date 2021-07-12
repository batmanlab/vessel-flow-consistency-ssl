'''
Given the 2D images, convert into a 3D volume that can be re-used.
'''
import SimpleITK as sitk
import numpy as np
from glob import glob
import os
from os import path as osp
from tqdm import tqdm

def getid(filename):
    return int(filename.split('/')[-1].split('_')[-1])

# Set main directory
maindir = '/ocean/projects/asc170022p/rohit33/Liver/'
for r, dirs, files in os.walk(maindir):
    dirs = map(lambda x: osp.join(r, x), dirs)
    break

def load(file):
    return sitk.GetArrayFromImage(sitk.ReadImage(file))

def save(img, file):
    img = sitk.GetImageFromArray(img)
    sitk.WriteImage(img, file)
    print("Saved to {}".format(file))


# For each directory, collect all images
for d in tqdm(dirs):
    imagelist = sorted(glob(osp.join(d, 'PATIENT_DICOM/image_*')), key=getid)
    #maskdir = sorted(glob(osp.join(d, 'MASKS_DICOM/liver/image_*')), key=getid)
    maskfiles = dict()
    for subdir in glob(osp.join(d, 'MASKS_DICOM/*')):
        files = sorted(glob(osp.join(subdir, 'image_*')), key=getid)
        maskfiles[subdir.split('/')[-1]] = files

    # Compile all images
    numslices = len(imagelist)
    image = []
    masks = dict()

    for k in maskfiles.keys():
        if not any([x in k for x in ['vein', 'liver', 'venous']]):
            continue
        masks[k] = []

    # Load image
    for imgfile in imagelist:
        img = load(imgfile)
        image.append(img)
    image = np.array(image).squeeze()
    #print(image.shape)

    # Load masks
    for maskid, files in maskfiles.items():
        if not any([x in maskid for x in ['vein', 'liver', 'venous']]):
            continue

        for file in files:
            maskimg = load(file)
            masks[maskid].append(maskimg)

    for maskid, imgs in masks.items():
        masks[maskid] = np.array(imgs).squeeze().astype(int)

    # Using livermask, mask it all
    livermask = masks['liver']
    livermask = 1.0*livermask/livermask.max()

    z, y, x = np.where(livermask > 0)
    zmin, zmax = z.min(), z.max() + 1
    ymin, ymax = y.min(), y.max() + 1
    xmin, xmax = x.min(), x.max() + 1

    # Crop
    croppedmask = livermask[zmin:zmax, ymin:ymax, xmin:xmax]
    image = image[zmin:zmax, ymin:ymax, xmin:xmax]
    image = image - image.min()
    image = image * croppedmask

    # Save
    save(image, osp.join(d, 'image.nii.gz'))
    save(croppedmask, osp.join(d, 'livermask.nii.gz'))
    for maskid, img in masks.items():
        #print(img.shape, maskid)
        cropimg = img[zmin:zmax, ymin:ymax, xmin:xmax] + 0
        cropimg = cropimg * croppedmask
        save(cropimg, osp.join(d, '{}.nii.gz'.format(maskid)))


