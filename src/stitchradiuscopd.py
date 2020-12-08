import pickle as pkl
import torch
from tqdm import tqdm
import torch
import numpy as np
import argparse
import SimpleITK as sitk
from data_loader.copddataset import COPDDataset

parser = argparse.ArgumentParser()
parser.add_argument("--patientID", type=str, required=True)

GAP=48
PATCH=64

def main():
    args = parser.parse_args()
    pid = args.patientID
    # Get dataset and vesselness
    ds = COPDDataset('/pghbio/dbmi/batmanlab/rohit33/COPD/', train=False, patientIDs=[pid], minibatch=1)
    filename = "/pghbio/dbmi/batmanlab/rohit33/copd_{}_analysis_3d.pkl".format(pid)
    with open(filename, 'rb') as fi:
        data = torch.load(fi)
    print(len(ds))
    print(len(data))

    # Get vesselness
    hmax, wmax, dmax = ds.total_patches[0][1:]
    hmax = hmax*GAP + PATCH
    wmax = wmax*GAP + PATCH
    dmax = dmax*GAP + PATCH
    # Vesselness and count
    radius = np.zeros((hmax, wmax, dmax))
    count = np.zeros((hmax, wmax, dmax))
    # Vesselness
    for i in tqdm(range(len(ds))):
        inp = ds[i] # input

        # Radius
        vessel = data[i]['vessel']  # Stored vessel
        vessel = torch.norm(vessel, dim=1).numpy().squeeze()

        _, h, w, d = inp['identifiers'].squeeze()
        radius[GAP*h:GAP*h + PATCH, GAP*w:GAP*w + PATCH, GAP*d:GAP*d + PATCH] += vessel
        count[GAP*h:GAP*h + PATCH, GAP*w:GAP*w + PATCH, GAP*d:GAP*d + PATCH] += 1


    outfilepath = "/pghbio/dbmi/batmanlab/rohit33/radius_{}.nii.gz".format(pid)
    outimg = radius/np.maximum(count, 1)
    sitk.WriteImage(sitk.GetImageFromArray(outimg), outfilepath)



if __name__ == '__main__':
    main()
