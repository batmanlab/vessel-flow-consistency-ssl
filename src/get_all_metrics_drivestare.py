'''
Use this script to generate threshold from the training set, and then use the test set to get metrics based on this threshold
'''
import torch
import pickle as pkl
import numpy as np
from tqdm import tqdm
import os
import sys
from data_loader.datasets import DriveDataset, StareDataset
from skimage.filters import frangi
from sklearn import metrics
from matplotlib import pyplot as plt
import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='frangi')
parser.add_argument('--dir', type=str, default='.')
parser.add_argument('--dataset', type=str, default='drive')
parser.add_argument('--threshold', type=float, default=None)

def frangi_vesselness(img, i):
    ves = frangi(img, np.linspace(1, 5, 5), black_ridges=True).astype(np.float32)
    return ves


def vesselness_file(filename):
    with open(filename, 'rb') as fi:
        data = pkl.load(fi)
    def v(img, i):
        V = data[i, 0]
        return V
    return v


def AUC(ves, G, num=1000):
    v = (ves - ves.min())/(ves.max() - ves.min())
    gt = G.astype(int)
    fpr, tpr, thres = metrics.roc_curve(gt.reshape(-1), v.reshape(-1), pos_label=1)
    return metrics.auc(fpr, tpr)


def multiply_mask(ves, mask):
    m = 0
    y, x = np.where(mask < 0.5)
    ves[y, x] = m
    return ves - m


def dice_score(a, b):
    num = (2*a*b).mean()
    den = a.mean() + b.mean()
    return num/den


def get_best_dice_threshold(ves, gt, thres, step=10):
    dicevals = []
    # Take dice values
    for t in thres[::step]:
        v = (ves >= t).astype(float)
        d = dice_score(v, gt.astype(float))
        dicevals.append(d)
    idx = np.argmax(dicevals)
    return thres[step*idx]


def compute_threshold(ds, gt, args):
    '''
    From the dataset, determine threshold to maximize Dice
    '''
    if args.method == 'frangi':
        vfunc = frangi_vesselness
    else:
        vfunc = vesselness_file(os.path.join(args.dir, 'train_vesselness.pkl'))

    Thres = []
    for i in tqdm(range(len(ds))):
        # if i == 5:
            # break
        #if i%5 == 0:
            #print("Index {}".format(i))
        img = ds[i]['image'][0].data.cpu().numpy()
        lab = (gt[i]['image'][0].data.cpu().numpy() >= 0.5).astype(int)
        mask = ds[i]['mask'][0].data.cpu().numpy()

        ves = vfunc(img, i)
        fpr, tpr, thres = metrics.roc_curve(lab.reshape(-1), ves.reshape(-1), pos_label=1)
        _bestthres = get_best_dice_threshold(ves, lab, thres)
        Thres.append(_bestthres)
    return np.mean(Thres)


def accuracy(v, gt):
    return (v == gt).astype(float).mean()

def specificity(v, gt):
    tn = ((v == 0)&(gt == 0)).mean()
    fp = ((v == 1)&(gt == 0)).mean()
    return tn/(tn + fp)

def sensitivity(v, gt):
    tp = ((v == 1)&(gt == 1)).mean()
    fn = ((v == 0)&(gt == 1)).mean()
    return tp/(tp + fn)


def print_all_test_metrics(ds, gt, args, threshold):
    # Print things like accuracy, AUC, etc
    if args.method == 'frangi':
        vfunc = frangi_vesselness
    else:
        vfunc = vesselness_file(os.path.join(args.dir, 'test_vesselness.pkl'))

    auc = []
    dice = []
    acc = []
    spec = []
    sens = []

    for i in range(len(ds)):
        img = ds[i]['image'][0].data.cpu().numpy()
        lab = (gt[i]['image'][0].data.cpu().numpy() > 0.5).astype(float)
        mask = ds[i]['mask'][0].data.cpu().numpy()
        # Get vesselness
        ves = vfunc(img, i)
        ves = multiply_mask(ves, mask)
        vthres = (ves >= threshold).astype(float)
        # Get metrics
        auc.append(AUC(ves, lab))
        acc.append(accuracy(vthres, lab))
        dice.append(dice_score(vthres, lab))
        spec.append(specificity(vthres, lab))
        sens.append(sensitivity(vthres, lab))

    print("Method: {}".format(args.method))
    print("AUC: {:.5f} , Acc: {:.5f} , Dice: {:.5f} , Sensitivity: {:.5f} , Specificity: {:.5f}".format(
              np.mean(auc), np.mean(acc), np.mean(dice), np.mean(sens), np.mean(spec)
        ))




def main():
    args = parser.parse_args()
    # Main function here
    traindataset = DriveDataset( "/pghbio/dbmi/batmanlab/rohit33/DRIVE/", train=True, augment=False)
    traingtdataset = DriveDataset( "/pghbio/dbmi/batmanlab/rohit33/DRIVE/", train=True, toy=True, augment=False)

    if args.dataset == 'drive':
        testdataset = DriveDataset( "/pghbio/dbmi/batmanlab/rohit33/DRIVE/", train=False, augment=False)
        testgtdataset = DriveDataset( "/pghbio/dbmi/batmanlab/rohit33/DRIVE/", train=False, toy=True, augment=False)
    else:
        testdataset = StareDataset( "/pghbio/dbmi/batmanlab/rohit33/STARE/", train=False, augment=False)
        testgtdataset = StareDataset( "/pghbio/dbmi/batmanlab/rohit33/STARE/", train=False, toy=True, augment=False)

    if args.threshold is None:
        threshold = compute_threshold(traindataset, traingtdataset, args)
    else:
        threshold = args.threshold

    # Compute all metrics
    print(threshold)
    print_all_test_metrics(testdataset, testgtdataset, args, threshold)



if __name__ == '__main__':
    main()