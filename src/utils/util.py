import json
import os
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import torch
from numpy import pi
import numpy as np
from matplotlib import pyplot as plt
import io
from PIL import Image
from skimage.color import hsv2rgb
from skimage import color
from model.loss import *
import string

ALPHABET = np.array(list(string.ascii_lowercase))

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


# overlay a quiver
def overlay_quiver(Img, flow, scale=2, normalize=True):
    # Given image of size [B, 1, H, W] and flow of size [B, 2, H, W]
    # output a quiver plot
    B, _, H, W = Img.shape
    hmin = np.random.randint(H-64+1)
    hmax = hmin + 64
    wmin = np.random.randint(W-64+1)
    wmax = wmin + 64
    # Crop image
    img = Img[:, :, hmin:hmax, wmin:wmax]
    vx = flow[:, 0, hmin:hmax, wmin:wmax].data.cpu().numpy()
    vy = flow[:, 1, hmin:hmax, wmin:wmax].data.cpu().numpy()
    if normalize:
        norm = np.sqrt(vx**2 + vy**2 + 1e-20)
        vx = vx / norm
        vy = vy / norm

    x = np.arange(64)
    y = np.arange(64)
    xx, yy = np.meshgrid(x, y)
    xx = xx[::scale, ::scale]
    yy = yy[::scale, ::scale]

    # For each image, add quiver plot
    images = []
    for i in range(B):
        _img = (img[i, 0].data.cpu().numpy())
        _vx = (vx[i])[::scale, ::scale]
        _vy = (-vy[i])[::scale, ::scale]
        if not plt.fignum_exists(1234):
            plt.figure(figsize=(16, 16), num=1234)
        plt.clf()
        plt.imshow(_img, 'gray')
        plt.quiver(xx, yy, _vx, _vy, color='lightgreen')
        plt.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        # load it back from file
        _tmpimg = Image.open(buf)
        _tmpimg = (np.array(_tmpimg)[:,:,:3]).transpose(2, 0, 1)[None]
        images.append(_tmpimg)
        buf.close()
    # images
    images = np.concatenate(images, 0)
    images = torch.FloatTensor(images)
    return images


# Other functions for vessels
def dir2flow_2d(flow, ret_mag=False):
    ''' Given a [B, 2, H, W] flow vector, convert into hsv and then to rgb '''
    B, C, H, W = flow.shape
    vx = flow[:, 0].data.cpu()
    vy = flow[:, 1].data.cpu()
    mag = torch.sqrt((vx**2 + vy**2) + 1e-100)
    ang = torch.atan2(vy, vx) % (2*pi)
    # normalize magnitude and angle to [0, 1]
    mag = mag / mag.max()
    ang = ang / (2*pi)
    if ret_mag:
        out = mag[:, None]
        return out
    else:
        # Given these two, get hsv first
        hsv = torch.FloatTensor(torch.zeros(B, H, W, 3))
        hsv[..., 0] = ang
        hsv[..., 1] = mag
        hsv[..., 2] = 1
        # Convert to rgb
        rgb = torch.FloatTensor(torch.zeros(B, H, W, 3))
        for i in range(B):
            hsv_i = hsv[i].numpy()
            rgb_i = hsv2rgb(hsv_i)
            rgb[i] = torch.FloatTensor(rgb_i)
        return rgb.permute(0, 3, 1, 2)

def v2transpose_vesselness(image, ves, nsample=20, vtype='light', mask=None, percentile=100, is_crosscorr=False):
    ves2 = ves*0
    ves2[:, 0] = -ves[:, 1]
    ves2[:, 1] = ves[:, 0]
    #return v2vesselness(image, ves2, nsample, vtype, mask, percentile, is_crosscorr)
    # Calculate the std deviation here
    response = []
    for s in np.linspace(-1, 1, nsample):
        i_val = resample_from_flow_2d(image, s*ves2)
        response.append(i_val.detach()[:, None])
    response = torch.cat(response, 1)
    v = response.std(1)
    return v


# Give an image and vessel overlay
def overlay(image, ves, alpha=0.4):
    # Given images of size [B, 1, H, W] and vesselness [B, 1, H, W]
    B, C, H, W = ves.shape
    img = (image - image.min())/(image.max() - image.min())
    # Init colormap image
    #cimg = torch.FloatTensor(np.zeros((B, 3, H, W)))
    cimg = []
    cimg_converter = plt.get_cmap('jet')
    for i in range(B):
        v = np.abs(ves[i, 0].detach().numpy()) + 0
        v = (v - v.min()) / (v.max() - v.min() + 1e-10)
        cv = cimg_converter(v)[:, :, :-1]       # 3
        ci = img[i, 0].detach().numpy()         # 1
        # Given cv of size [H, W, 3] and [H, W, 1]
        #f_img = (1-alpha)*ci[..., None] + alpha*cv
        #f_img = ci[..., None] * cv
        if not plt.fignum_exists(1234):
            plt.figure(figsize=(16, 16), num=1234)
        plt.clf()
        plt.imshow(ci, 'gray')
        plt.imshow(cv, alpha=alpha)
        plt.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        _tmpimg = Image.open(buf)
        _tmpimg = (np.array(_tmpimg)[:,:,:3]).transpose(2, 0, 1)[None]
        cimg.append(_tmpimg)
        buf.close()

        '''
        hsv_vessel = color.rgb2hsv(cv)
        hsv_img = color.rgb2hsv(color.gray2rgb(ci))
        # replace hsv of img with hsv of vessel
        hsv_img[..., 0] = hsv_vessel[..., 0]
        hsv_img[..., 1] = hsv_vessel[..., 1] * alpha
        f_img = color.hsv2rgb(hsv_img)
        '''
        #cimg[i] = torch.FloatTensor(f_img.transpose(2, 0, 1))
    cimg = np.concatenate(cimg, 0)
    cimg = torch.FloatTensor(cimg)
    # img is in [0, 1]  --> [B, 1, H, W]
    # cimg is in [0, 1] --> [B, 3, H, W]
    return cimg

