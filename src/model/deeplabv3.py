import torch
from torch import nn
from torch.nn import functional as F

from torchvision import models


class DeepLabV3(nn.Module):

    def __init__(self, inp_channels=1, out_channels=4, vessel_scale_factor=16, bilinear=True):
        super().__init__()
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.vessel_scale_factor = vessel_scale_factor

        self.initconv = nn.Conv2d(self.inp_channels, 3, kernel_size=1, )
        self.net = models.segmentation.deeplabv3_resnet101(pretrained=False)

        # for the deeplab
        #def __init__(self, backbone, classifier, aux_classifier=None):

    def forward(self, x1):
        x = x1['image']
        x = self.initconv(x)
        allout = self.net(x)

        # Get vessel outputs and reconstruction
        out = allout['out'][:, :self.out_channels*3//2]
        recon = allout['out'][:, -self.inp_channels:]

        # convert vessel into scale and direction
        chan = int(self.out_channels//2)
        B, _, H, W = out.shape
        ves = []
        for i in range(chan):
            direction = out[:, 2*i:2*i+2]
            scale = (out[:, 2*chan + i])[:, None]
            scale = self.vessel_scale_factor * torch.sigmoid(scale)
            norm = torch.sqrt(1e-10 + (direction**2).sum(1))[:, None]
            ves.append(direction / norm * scale)
        out = torch.cat(ves, 1)

        return {
            'recon': recon,
            'vessel': out,
        }



