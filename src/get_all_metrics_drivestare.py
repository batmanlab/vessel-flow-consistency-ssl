'''
Use this script to generate threshold from the training set, 
and then use the test set to get metrics based on this threshold

Use for Frangi, Sato, Meijering, Hessian filters and our method 
(saved in a file) for DRIVE and STARE dataset
'''
import torch
import pickle as pkl
import numpy as np
from tqdm import tqdm
import os
import sys
import xml.etree.ElementTree as ET
from PIL import Image
from data_loader.datasets import DriveDataset, StareDataset
from skimage.filters import frangi, sato, meijering, hessian
from scipy.ndimage.morphology import binary_dilation
from sklearn import metrics
from matplotlib import pyplot as plt
import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='frangi')
parser.add_argument('--dir', type=str, default='.')
parser.add_argument('--dataset', type=str, default='drive')
parser.add_argument('--threshold', type=float, default=None)
parser.add_argument('--mode', type=str, default='test')
parser.add_argument('--fold', type=int, default=-1)

def get_bbox_anno(filename):
    '''
    Script to get out the rectangle parameters given a filename in XML format
    '''
    tree = ET.parse(filename)
    root = tree.getroot()
    objects = list(filter(lambda x: x.tag == 'object', list(root)))
    bboxobj = list(map(lambda x: list(x), objects))
    bboxobj = [x for sublist in bboxobj for x in sublist]
    bboxobj = list(filter(lambda x: x.tag == 'bndbox', bboxobj))

    bboxes = []
    for box in bboxobj:
        xy = list(box)
        d = dict()
        for _xy in xy:
            d[_xy.tag] = int(_xy.text)
        bbox = [d['xmin'], d['xmax'], d['ymin'], d['ymax']]
        bboxes.append(bbox)
    return bboxes

def frangi_vesselness(img, i):
    ves = frangi(img, sigmas=np.linspace(1, 5, 5), black_ridges=True).astype(np.float32)
    return ves

def meijering_vesselness(img, i):
    im  = img
    ves = meijering(im, sigmas=np.linspace(1, 5, 5), black_ridges=True).astype(np.float32)
    return ves

def sato_vesselness(img, i):
    ves = sato(img, sigmas=np.linspace(1, 5, 5), black_ridges=True).astype(np.float32)
    return ves

def hessian_vesselness(img, i):
    ves = hessian(img, sigmas=np.linspace(1, 5, 5), black_ridges=True).astype(np.float32)
    return ves

def vesselness_file(filename):
    '''Given a filename, open the vesselness which is stored as a tensor of size [B, 1, H, W]'''
    with open(filename, 'rb') as fi:
        data = pkl.load(fi)
    print("Using output from {}".format(filename))
    def v(img, i):
        V = data[i, 0]
        return V
    return v

def AUC(ves, G, num=None):
    '''Calculate AUC from images'''
    #TODO: remove `num` parameter without breaking code
    v = (ves - ves.min())/(ves.max() - ves.min())
    gt = G.astype(int)
    fpr, tpr, thres = metrics.roc_curve(gt.reshape(-1), v.reshape(-1), pos_label=1)
    return metrics.auc(fpr, tpr)

def multiply_mask(ves, mask, takemin=True):
    ''' Multiply vesselness with given mask of the image (to prevent edge artefacts) '''
    if takemin:
        m = ves.min()
    else:
        m = 0
    y, x = np.where(mask < 0.5)
    ves[y, x] = m
    return ves - m

def dice_score(a, b):
    '''Calculate dice score b/w two images `a` and `b` '''
    if a.size == 0:
        return np.nan

    num = (2*a*b).mean()
    den = a.mean() + b.mean() + 1e-100
    return num/den

def get_best_dice_threshold(ves, gt, thres, step=10):
    '''Given a sequence of threshold values, pick the one which gives best dice score
    This code should be run on `ves` and `gt` from the validation set to choose
    the best dice score
    '''
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
    From the dataset, determine threshold to maximize Dice over the dataset
    ds: Data loader
    gt: Ground truth data loader
    args: arguments for choosing vesselness method, fold (in k-fold validation)
    '''
    if args.method == 'frangi':
        vfunc = frangi_vesselness
    elif args.method == 'sato':
        vfunc = sato_vesselness
    elif args.method == 'hessian':
        vfunc = hessian_vesselness
    elif args.method == 'meijering':
        vfunc = meijering_vesselness
    elif args.method == 'ours':
        vfunc = vesselness_file(os.path.join(args.dir, 'train_vesselness.pkl'))
    else:
        raise NotImplementedError

    Thres = []
    fold = args.fold
    for i in tqdm(range(len(ds))):
        if args.fold != -1:
            if i >= 4*fold and i - 4*fold < 4:
                pass
            else:
                continue
        img = ds[i]['image'][0].data.cpu().numpy()
        lab = (gt[i]['image'][0].data.cpu().numpy() >= 0.5).astype(float)
        mask = ds[i]['mask'][0].data.cpu().numpy()

        ves = vfunc(img, i)
        fpr, tpr, thres = metrics.roc_curve(lab.reshape(-1), ves.reshape(-1), pos_label=1)
        _bestthres = get_best_dice_threshold(ves, lab, thres)
        Thres.append(_bestthres)
    return np.mean(Thres)


def accuracy(v, gt):
    return (v == gt).astype(float).mean()

def localaccuracy(v, gt):
    '''Accuracy only around a neighborhood region of the ground truth vessel
    This metric helps to see how accurate the method is at the pixel level
    '''
    gtdilate = binary_dilation(gt, np.ones((3, 3)))
    y, x = np.where(gtdilate > 0)
    acc = accuracy(v[y, x], gt[y, x])
    return acc

def specificity(v, gt):
    tn = ((v == 0)&(gt == 0)).mean()
    fp = ((v == 1)&(gt == 0)).mean()
    return tn/(tn + fp)

def sensitivity(v, gt):
    tp = ((v == 1)&(gt == 1)).mean()
    fn = ((v == 0)&(gt == 1)).mean()
    return tp/(tp + fn)


def get_anno_metrics(ds, gt, args, threshold):
    ''' Get all metrics for the given method (at bifurcation locations only)
    This function gets the xml annotation file, and gets metrics like DICE, AUC,
    and accuracy.
    '''
    if args.method == 'frangi':
        vfunc = frangi_vesselness
    elif args.method == 'sato':
        vfunc = sato_vesselness
    elif args.method == 'hessian':
        vfunc = hessian_vesselness
    elif args.method == 'meijering':
        vfunc = meijering_vesselness
    elif args.method == 'ours':
        vfunc = vesselness_file(os.path.join(args.dir, 'test_vesselness.pkl'))

    dice = []
    auc = []
    acc = []
    for i in range(10):
        annoname = "/pghbio/dbmi/batmanlab/rohit33/DRIVE/test/branchannotations/{:02d}_manual1.xml".format(i+1)
        #print(annoname)
        bboxes = get_bbox_anno(annoname)
        ## Get dataloader
        img = ds[i]['image'][0].data.cpu().numpy()
        lab = (gt[i]['image'][0].data.cpu().numpy() >= 0.5).astype(float)
        mask = ds[i]['mask'][0].data.cpu().numpy()
        # Get vesselness
        ves = vfunc(img, i)
        ves = multiply_mask(ves, mask, takemin=args.method != 'ours')
        vthres = (ves >= threshold).astype(float)
        # Take bounding boxes
        for xmin, xmax, ymin, ymax in bboxes:
            try:
                vpred = vthres[ymin:ymax+1, xmin:xmax+1]
                vgt = lab[ymin:ymax+1, xmin:xmax+1]
                if vpred.size == 0:
                    continue
                if vgt.astype(float).mean() < 0.25:
                    continue
                _dice = dice_score(vpred, vgt)
                _auc = AUC(ves[ymin:ymax+1, xmin:xmax+1], vgt)
                _acc = (vpred == vgt).astype(float).mean()
                if not np.isnan(_dice):
                    dice.append(_dice)
                if not np.isnan(_auc):
                    auc.append(_auc)
                if not np.isnan(_acc):
                    acc.append(_acc*100)
            except:
                pass
    print("Dice Mean: {:.2f}, Std: {:.2f}".format(np.mean(dice), np.std(dice)))
    print("AUC Mean: {:.2f}, Std: {:.2f}".format(np.mean(auc), np.std(auc)))
    print("Acc Mean: {:.2f}, Std: {:.2f}".format(np.mean(acc), np.std(acc)))


def get_image_difference(vthres, lab):
    '''Get a color coded difference image between predicted and ground-truth 
    vessel binary mask. Different colors are used for false positives, and false negatives.
    '''
    # Get difference
    H, W = vthres.shape
    diff = np.zeros((H, W, 3))
    # Color tp pixels white
    tp = (vthres == 1)&(lab == 1)
    y, x = np.where(tp)
    diff[y, x, :] = 1.0
    # Color fp pixels red
    fp = (vthres == 1)&(lab == 0)
    y, x = np.where(fp)
    diff[y, x, 0] = 1.0
    # Color fn pixels green
    fn = (vthres == 0)&(lab == 1)
    y, x = np.where(fn)
    diff[y, x, 1] = 1.0
    diff = (diff * 255).astype(np.uint8)
    # Get image
    im = Image.fromarray(diff)
    return im

def print_all_test_metrics(ds, gt, args, threshold):
    ''' Self-explanatory function name
    '''
    # Print things like accuracy, AUC, etc
    if args.method == 'frangi':
        vfunc = frangi_vesselness
    elif args.method == 'sato':
        vfunc = sato_vesselness
    elif args.method == 'hessian':
        vfunc = hessian_vesselness
    elif args.method == 'meijering':
        vfunc = meijering_vesselness
    elif args.method == 'ours':
        vfunc = vesselness_file(os.path.join(args.dir, 'test_vesselness.pkl'))
    else:
        raise NotImplementedError

    auc = []
    dice = []
    acc = []
    spec = []
    sens = []
    localacc = []

    os.makedirs("drive{}test".format(args.method), exist_ok=True)
    for i in range(len(ds)):
        img = ds[i]['image'][0].data.cpu().numpy()
        lab = (gt[i]['image'][0].data.cpu().numpy() >= 0.5).astype(float)
        mask = ds[i]['mask'][0].data.cpu().numpy()
        # Get vesselness
        ves = vfunc(img, i)
        ves = multiply_mask(ves, mask, takemin=args.method != 'ours')
        vthres = (ves >= threshold).astype(float)

        # Difference image
        diffim = get_image_difference(vthres, lab)
        diffim.save("drive{}test/diff_{}.png".format(args.method, ds.images[i].split('/')[-1]))
        # save it
        saveim = Image.fromarray((np.tile(255*vthres[:, :, None], (1, 1, 3))).astype(np.uint8))
        saveim.save("drive{}test/{}.png".format(args.method, ds.images[i].split('/')[-1]))

        # Get metrics
        auc.append(AUC(ves, lab))
        acc.append(accuracy(vthres, lab))
        dice.append(dice_score(vthres, lab))
        spec.append(specificity(vthres, lab))
        sens.append(sensitivity(vthres, lab))
        localacc.append(localaccuracy(vthres, lab))

    print("Method: {}".format(args.method))
    print("AUC: {:.5f} , Acc: {:.5f} , Dice: {:.5f} , Sensitivity: {:.5f} , Specificity: {:.5f}, Local accuracy: {:.5f}".format(
              np.mean(auc), 100*np.mean(acc), np.mean(dice), np.mean(sens), np.mean(spec), 100*np.mean(localacc)
        ))
    print("Stddeviation:")
    print("AUC: {:.5f} , Acc: {:.5f} , Dice: {:.5f} , Sensitivity: {:.5f} , Specificity: {:.5f}, Local accuracy: {:.5f}".format(
              np.std(auc), 100*np.std(acc), np.std(dice), np.std(sens), np.std(spec), 100*np.std(localacc)
        ))

''' Main function, first initialize the datasets, then either compute threshold 
or use predefined threshold (saved from previous runs), and using that threshold
calculate metrics for the test set.
'''
def main():
    args = parser.parse_args()
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
    if args.mode == 'test':
        print_all_test_metrics(testdataset, testgtdataset, args, threshold)
    elif args.mode == 'anno':
        get_anno_metrics(testdataset, testgtdataset, args, threshold)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()
