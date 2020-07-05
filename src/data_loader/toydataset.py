import os
import numpy as np
from numpy import pi, sin, cos, exp
from os import path as osp
import torch
from torch.utils.data import Dataset
from PIL import Image

class ToyStrLines(Dataset):

    def __init__(self, img_size, train=True):
        self.img_size = img_size


    def __len__(self):
        return (15 - 3 + 1) * 180

    def __getitem__(self, idx):
        thickness = idx // 180 + 3
        rot = idx % 180
        img = self.create_straight_vessel(self.img_size, thickness, rot)
        img = img[None]
        return {
            'image': torch.FloatTensor(img),
            'mask' : torch.FloatTensor(img),
        }


    def create_straight_vessel(self, img_size=32, vessel_thickness=5, rotate=0):
        '''
        Given an image and vessel size, and rotation of vessel
        rotation is in degrees
        '''
        img = np.zeros((img_size, img_size))
        C = img_size // 2
        x = np.arange(img_size) - C
        xx, yy = np.meshgrid(x, x)
        # Get angle
        theta = rotate / 180 * pi
        sint = sin(theta)
        cost = cos(theta)
        # Get distance of each point from
        dist = np.abs(sint*xx - cost*yy)
        vessel = (dist < vessel_thickness//2).astype(int)
        return vessel



class ToySlantLines(Dataset):

    def __init__(self, img_size, train=True):
        self.img_size = img_size
        self.maxlocalrot = 6

    def __len__(self):
        return (15 - 5 + 1) * 180 * self.maxlocalrot

    def __getitem__(self, idx):
        # get max rotation
        maxrot = idx % self.maxlocalrot
        idx = idx // self.maxlocalrot
        # Get base thickness and rotation
        thickness = idx // 180 + 5
        rot = idx % 180
        local_rot = np.random.rand() + maxrot
        img = self.create_slant_vessel(self.img_size, thickness, rot, local_rotate=local_rot)
        img = img[None]
        return {
            'image': torch.FloatTensor(img),
            'mask' : torch.FloatTensor(img),
        }


    def create_slant_vessel(self, img_size=32, vessel_thickness=5, rotate=0, local_rotate=0):
        '''
        Given an image and vessel size, and rotation of vessel
        rotation is in degrees
        '''
        img = np.zeros((img_size, img_size))
        C = img_size // 2
        x = np.arange(img_size) - C
        xx, yy = np.meshgrid(x, x)
        # Get angle
        theta = rotate / 180 * pi
        theta2 = local_rotate / 180 * pi
        # Get distance of each point from
        dist1 = sin(theta + theta2)*xx - cos(theta + theta2)*yy
        dist2 = sin(theta - theta2)*xx - cos(theta - theta2)*yy
        vessel1 = (dist1 < vessel_thickness//2).astype(int)
        vessel2 = (dist2 > -vessel_thickness//2).astype(int)
        vessel = vessel1*vessel2
        return vessel


class ToyCurvedLines(Dataset):

    def __init__(self, img_size, train=True):
        self.img_size = img_size

    def __len__(self):
        return 180 * 9 * 10

    def __getitem__(self, idx):
        rot = (idx % 180) * 2
        idx = idx // 180
        # Get curvature
        curv = (idx % 9)*0.1 + 0.1
        idx = idx // 9
        # Get size
        size = idx + 5
        img = self.create_curved_vessel(self.img_size, vessel_thickness=size, curvature=curv, rotate=rot)
        img = img[None]
        return {
            'image': torch.FloatTensor(img),
            'mask' : torch.FloatTensor(img)
        }

    def image_sample(self, Img, x, y, cval=0):
        ''' Sample numpy image '''
        shape = list(Img.shape)
        shape[0] += 1
        shape[1] += 1
        img = np.zeros(shape)
        img[:-1, :-1] = Img + 0
        assert x.shape == y.shape, 'Coordinates need to have the same shape'
        # sample img given fractional x and y
        H, W = Img.shape[:2]
        if len(img.shape) == 3:
            C = img.shape[2]
        else:
            C = 1
        outimg = np.zeros((*x.shape, C)) + cval
        # Get valid coordinates
        valididx = (0 <= x)*(x <= W-1)*(0 <= y)*(y <= H-1)
        nullidx = ~valididx
        # for all valid coordinates, get the values of intensity
        xf = np.floor(x[valididx]).astype(int)
        xc = xf + 1
        yf = np.floor(y[valididx]).astype(int)
        yc = yf + 1
        _x, _y = x[valididx], y[valididx]
        # find values
        img_sampled = img[yf, xf]*(yc - _y)*(xc - _x) \
                    + img[yf, xc]*(yc - _y)*(_x - xf) \
                    + img[yc, xf]*(_y - yf)*(xc - _x) \
                    + img[yc, xc]*(_y - yf)*(_x - xf)
        if C == 1:
            outimg[valididx, 0] = img_sampled.squeeze()
        else:
            outimg[valididx] = img_sampled
        return outimg.squeeze()

    def create_curved_vessel(self, img_size, vessel_thickness=5, curvature=1, rotate=0):
        ' Create a curved vessel'
        img = np.zeros((img_size, img_size))
        C = img_size // 2
        x = np.arange(img_size) - C
        xx, yy = np.meshgrid(x, x)
        # Get coordinate
        rad = C / curvature
        # Get center otherwise
        angle = rotate * pi / 180
        cx = cos(angle)*rad
        cy = sin(angle)*rad
        # Given these points, just return vessel
        vessel = exp(-np.abs((xx - cx)**2 + (yy - cy)**2 - rad**2)/2/(C)**2)
        # Determine intensity for thickness of vessel
        oy, ox = np.where((xx == 0)&(yy == 0))
        angles = np.linspace(0, 2*pi, 100)
        ay = oy + sin(angles)*vessel_thickness//2
        ax = ox + cos(angles)*vessel_thickness//2
        ivals = self.image_sample(vessel, ax, ay)
        thres = np.min(ivals)
        # Return vessel
        V = vessel > thres
        return V
