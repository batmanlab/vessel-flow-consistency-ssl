import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from base import BaseModel

class FCN32s(BaseModel):

    def __init__(self, inp_channels=3, out_channels=4):
        super(FCN32s, self).__init__()
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        # conv1
        self.conv1_1 = nn.Conv2d(inp_channels, 64, 3, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)

        # conv3
        self.conv3_1 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(128, self.out_channels, 1, padding=0)

        # conv4
        self.conv4_1 = nn.Conv2d(self.out_channels, 128, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)

        # conv5
        self.conv5_1 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)

        # fc (this is where we will get the vessel)
        self.fc1 = nn.Conv2d(128, inp_channels, 3, padding=1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def forward(self, x):
        h = x['image']
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        out = self.conv3_3(h)

        h = self.relu4_1(self.conv4_1(out))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.fc1(h)
        return {
            'recon': h,
            'vessel': out,
        }

