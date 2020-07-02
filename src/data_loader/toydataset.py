import os
import numpy as np
from numpy import pi, sin, cos
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

