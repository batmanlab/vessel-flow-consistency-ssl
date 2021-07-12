import argparse
import glob
from os import path as osp
import torch
import pickle as pkl
import numpy as np
from tqdm import tqdm
import os
import sys
from PIL import Image
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from sklearn import metrics
from matplotlib import pyplot as plt
import argparse
import cv2

modify = lambda x, y: x

parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='frangi')
parser.add_argument('--dir', type=str, default='/ocean/projects/asc170022p/rohit33/TubeTKoutput')
parser.add_argument('--threshold', type=float, default=None)
parser.add_argument('--mode', type=str, default='test')

def auc(gt, out):
    fpr, tpr, thres = metrics.roc_curve(gt.reshape(-1), out.reshape(-1), pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc

def sp(gt, v):
    tn = ((v == 0)&(gt == 0)).mean()
    fp = ((v == 1)&(gt == 0)).mean()
    return tn/(tn + fp)

def sc(gt, v):
    tp = ((v == 1)&(gt == 1)).mean()
    fn = ((v == 0)&(gt == 1)).mean()
    return tp/(tp + fn)

def get_metrics(ves, gt):
    tp = (ves==1)&(gt==1)
    fp = (ves==1)&(gt==0)
    fn = (ves==0)&(gt==1)

    tpr = tp.mean()/gt.mean()
    fpr = fp.mean()/(1 - gt).mean()
    fnr = fn.mean()/gt.mean()
    return tpr, fpr, fnr

def acc(a, b):
    return (a == b).mean()

def getid(filename):
    return int(filename.split('/')[-1].split('.')[0].split('_')[1])

def dilategt(gt):
    mask = np.zeros((3, 3, 3))
    mask[1] = 1
    mask[:, 1] = 1
    mask[:, :, 1] = 1
    return binary_dilation(gt, mask).astype(float)
    #return binary_dilation(gt, np.ones((2, 2, 2))).astype(float)
    #return gt

def loadgt(gtfile):
    gt = np.load(gtfile)
    gt = dilategt(gt)
    return gt

def erode(ves):
    mask = np.zeros((3, 3, 3))
    mask[1] = 1
    mask[:, 1] = 1
    mask[:, :, 1] = 1
    return binary_erosion(ves, mask).astype(float)

def dice(a, b):
    num = 2*(a*b).mean()
    den = a.mean() + b.mean()
    return num/den

def get_best_dice_threshold(ves, lab, thres):
    bestt = None
    bestd = -1
    N = int(len(thres)/100.0)
    for t in (thres[::N]):
        #d = dice(ves>=t , lab)
        vess = (ves>=t).astype(float)
        d = dice(vess, lab)
        if d > bestd:
            bestt = t
            bestd = d

    # Get vesselness and rates
    vesbin = (ves >= bestt).astype(float)
    tpr, fpr, fnr = get_metrics(vesbin, lab)

    print("Got best dice {:.4f} at threshold {}".format(bestd, bestt))
    print("TPR: {:.4f}, FPR: {:.4f}, FNR: {:.4f}".format(tpr, fpr, fnr))
    return bestt


def get_threshold(gtfiles, ofiles,):
    # Given list of ground truths and outputs, figure out (close to) optimal threshold
    thresvals = []
    for g, o in (list(zip(gtfiles, ofiles))):
        lab = loadgt(g)
        ves = modify(np.load(o), g)
        fpr, tpr, thres = metrics.roc_curve(lab.reshape(-1), ves.reshape(-1), pos_label=1)
        print(o)
        _bestthres = get_best_dice_threshold(ves, lab, thres)
        print()
        thresvals.append(_bestthres)

        #auc = metrics.auc(fpr, tpr)
        #thresvals.append(auc)
    return np.mean(thresvals)


def modifyours(img, gtfile):
    m = img.min()
    hull = np.load(gtfile.replace('gt', 'hull'))
    #print("Loading hull")
    img[hull == 0] = m
    #img[0] = m
    #img[-1] = m
    #img[:, 0, :] = m
    #img[:, -1, :] = m
    #img[:, :, 0] = m
    #img[:, :, -1] = m
    return img


def main():
    global modify
    args = parser.parse_args()
    #gtfiles = sorted(glob.glob(osp.join(args.dir, 'gt*npy')), key=getid)
    outputfiles = sorted(glob.glob(osp.join(args.dir, '{}*npy'.format(args.method))), key=getid)#[8:]
    gtfiles = [x.replace(args.method, 'gt') for x in outputfiles]
    #print(outputfiles[0], gtfiles[0])

    #if args.method == 'ours':
    modify = modifyours

    N = 42
    #auc = get_threshold(gtfiles[:N], outputfiles[:N])
    #print("AUC: {}".format(auc))

    if args.threshold is None:
        thres = get_threshold(gtfiles[:N], outputfiles[:N])
    else:
        thres = args.threshold
    print("Threshold: {}".format(thres))

    #alldice = []
    metrics = dict(dice=[], acc=[], sp=[], sc=[], auc=[])
    for g, o in tqdm(list(zip(gtfiles, outputfiles))):
    #for g, o in tqdm(list(zip(gtfiles[N:], outputfiles[N:]))):
        gt = loadgt(g)
        outv = modify(np.load(o), g)
        out = (outv >= thres).astype(float)
        metrics['dice'].append(dice(gt, out))
        metrics['acc'].append(100*acc(gt, out))
        metrics['sp'].append(sp(gt, out))
        metrics['sc'].append(sc(gt, out))
        metrics['auc'].append(auc(gt, outv))

    for k, v in metrics.items():
        mean = np.mean(v)
        std = np.std(v)
        print("{} mean: {:.2f} std: {:.2f}".format(k, mean, std))


if __name__ == "__main__":
    main()
