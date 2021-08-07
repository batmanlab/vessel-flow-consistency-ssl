'''
Use this script to generate threshold from the training set, 
and then use the test set to get metrics based on this threshold

Compare Frangi and our method for VESSEL12 dataset
'''
import argparse
import glob
import pandas as pd
from os import path as osp
import torch
import pickle as pkl
import numpy as np
from tqdm import tqdm
import os
import sys
from PIL import Image
from scipy.ndimage.morphology import binary_dilation
from sklearn import metrics
from matplotlib import pyplot as plt
import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='frangi')
parser.add_argument('--dir', type=str, default='/ocean/projects/asc170022p/rohit33/VESSEL12output')
parser.add_argument('--threshold', type=float, default=None)
parser.add_argument('--subdir', type=str, default='')
parser.add_argument('--mode', type=str, default='test')

def dice(a, b):
    num = 2*(a*b).mean()
    den = a.mean() + b.mean()
    return num/den

def accuracy(a, b):
    return (a == b).mean()

def sp(v, gt):
    tn = ((v == 0)&(gt == 0)).mean()
    fp = ((v == 1)&(gt == 0)).mean()
    return tn/(tn + fp)

def sc(v, gt):
    tp = ((v == 1)&(gt == 1)).mean()
    fn = ((v == 0)&(gt == 1)).mean()
    return tp/(tp + fn)

def ppv(v, gt):
    # Precision
    tp = ((v == 1)&(gt == 1)).mean()
    fp = ((v == 1)&(gt == 0)).mean()
    return tp/(tp + fp)

def auc(a, b):
    # a is the prediction, b is the binary ground truth
    fpr, tpr, thres = metrics.roc_curve(b.reshape(-1), a.reshape(-1), pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc

def get_best_thres(pred, label):
    bestt = None
    bestacc = -1
    for t in pred:
        pred_thres = (pred >= t).astype(float)
        #acc = (pred_thres == label).mean()
        acc = dice(pred_thres, label) + dice(1-pred_thres, 1-label)
        if acc > bestacc:
            bestacc = acc
            bestt = t
    return bestt, bestacc

def main():
    args = parser.parse_args()
    outputfiles = sorted(glob.glob(osp.join(args.dir, '{}{}*'.format(args.subdir, args.method))), )
    gtfiles = sorted(glob.glob(osp.join(args.dir, '*csv')), )

    N1 = min(len(gtfiles), len(outputfiles))
    outputfiles = outputfiles[:N1]
    gtfiles = gtfiles[:N1]

    offset = 1
    if offset > 0:
        outputfiles = outputfiles[offset:] + outputfiles[:offset]
        gtfiles = gtfiles[offset:] + gtfiles[:offset]

    # Get threshold
    aucs = []
    N = 1
    bestthres = []

    for g, o in zip(gtfiles[:N], outputfiles[:N]):
        #print(g, o)
        csv = np.array(pd.read_csv(g, header=None))
        x, y, z, label = csv.T # gt
        # Load image
        img = np.load(o)
        pred = img[z, y, x]
        thres, acc = get_best_thres(pred, label)
        bestthres.append(thres)

    # Compute metrics
    metrics = dict(acc=[], auc=[], sp=[], sc=[], )

    for i, (g, o) in enumerate(zip(gtfiles[N:], outputfiles[N:])):
        print(g, o)
        metrics = dict(acc=[], auc=[], sp=[], sc=[], )
        csv = np.array(pd.read_csv(g, header=None))
        x, y, z, label = csv.T # gt
        # Load image
        img = np.load(o)
        pred = (img[z, y, x] >= bestthres[i+N])
        predv = img[z, y, x]
        # Save metrics
        metrics['acc'].append(100*accuracy(pred, label))
        #metrics['dice'].append(dice(pred, label))
        metrics['auc'].append(auc(predv, label))
        metrics['sp'].append(sp(pred, label))
        metrics['sc'].append(sc(pred, label))
        #metrics['ppv'].append(ppv(pred, label))
        for k, v in metrics.items():
            print("{} {:.4f} {:.4f}".format(k, np.mean(v), np.std(v)))


if __name__ == "__main__":
    main()
