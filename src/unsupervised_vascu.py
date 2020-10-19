import torch
from tqdm import tqdm
import pickle as pkl
import numpy as np
import sys
from data_loader.datasets import VascuDataset
from skimage.filters import frangi
from sklearn import metrics
from matplotlib import pyplot as plt
import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='frangi')
parser.add_argument('--file', type=str, default='test_vesselness_3d.pkl')
parser.add_argument('--sigma', type=float, required=True)

def frangi_vesselness(img, i):
    ves = frangi(img, np.linspace(1, 12, 12), black_ridges=False).astype(np.float32)
    return ves


def vesselness_file(filename):
    with open(filename, 'rb') as fi:
        data = pkl.load(fi)

    def v(img, i):
        V = data[i][0, 0]
        return V
    return v


def get_vesselness(args):
    ''' Get vesselness from method '''
    if args.method == 'frangi':
        return frangi_vesselness
    elif args.method == 'file':
        return vesselness_file(args.file)
    else:
        raise NotImplementedError


def AUC(ves, G, num=1000):
    v = (ves - ves.min())/(ves.max() - ves.min())
    gt = G.astype(int)
    fpr, tpr, thres = metrics.roc_curve(gt.reshape(-1), v.reshape(-1), pos_label=1)
    return metrics.auc(fpr, tpr)


def multiply_mask(ves, mask):
    m = ves.min()
    y, x = np.where(mask < 0.5)
    ves[y, x] = m
    return ves - m


def main():
    args = parser.parse_args()
    # Main function here
    dataset = VascuDataset( "/pghbio/dbmi/batmanlab/rohit33/VascuSynth/", train=False, sigma=args.sigma)

    vfunc = get_vesselness(args)
    all_auc = []
    print(len(dataset))
    for i in tqdm(range(len(dataset))):
        img = dataset[i]['image'][0].data.cpu().numpy()
        lab = (dataset[i]['gt'][0].data.cpu().numpy()).astype(int)
        # Get vesselness
        ves = vfunc(img, i)
        auc = AUC(ves, lab)
        all_auc.append(auc)
    print("Method: {}, mean AUC: {}, std AUC: {}".format(args.method, np.mean(all_auc), np.std(all_auc)**2))


if __name__ == '__main__':
    main()
