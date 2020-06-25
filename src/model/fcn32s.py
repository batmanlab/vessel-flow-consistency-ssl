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
        self.relu1_1 = nn.LeakyReLU()
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.LeakyReLU()
        self.bn1_2 = nn.BatchNorm2d(64)

        # conv2
        self.conv2_1 = nn.Conv2d(64, 32, 3, padding=1)
        self.relu2_1 = nn.LeakyReLU()
        self.bn2_1 = nn.BatchNorm2d(32)
        self.conv2_2 = nn.Conv2d(32, 32, 3, padding=1)
        self.relu2_2 = nn.LeakyReLU()
        self.bn2_2 = nn.BatchNorm2d(32)

        # conv3
        self.conv3_1 = nn.Conv2d(32, 32, 5, padding=2)
        self.relu3_1 = nn.LeakyReLU()
        self.bn3_1 = nn.BatchNorm2d(32)
        self.conv3_2 = nn.Conv2d(32, 32, 5, padding=2)
        self.relu3_2 = nn.LeakyReLU()
        self.bn3_2 = nn.BatchNorm2d(32)

        self.conv3_3 = nn.Conv2d(32, self.out_channels, 1, padding=0)

        # conv4
        self.conv4_1 = nn.Conv2d(self.out_channels, 32, 5, padding=2)
        self.relu4_1 = nn.LeakyReLU()
        self.bn4_1 = nn.BatchNorm2d(32)
        self.conv4_2 = nn.Conv2d(32, 32, 5, padding=2)
        self.relu4_2 = nn.LeakyReLU()
        self.bn4_2 = nn.BatchNorm2d(32)
        self.conv4_3 = nn.Conv2d(32, 32, 5, padding=2)
        self.relu4_3 = nn.LeakyReLU()
        self.bn4_3 = nn.BatchNorm2d(32)

        # conv5
        self.conv5_1 = nn.Conv2d(32, 32, 5, padding=2)
        self.relu5_1 = nn.LeakyReLU()
        self.bn5_1 = nn.BatchNorm2d(32)
        self.conv5_2 = nn.Conv2d(32, 32, 5, padding=2)
        self.relu5_2 = nn.LeakyReLU()
        self.bn5_2 = nn.BatchNorm2d(32)
        self.conv5_3 = nn.Conv2d(32, 32, 3, padding=1)
        self.relu5_3 = nn.LeakyReLU()
        self.bn5_3 = nn.BatchNorm2d(32)

        # fc (this is where we will get the vessel)
        self.fc1 = nn.Conv2d(32, inp_channels, 3, padding=1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if False and isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        h = x['image']
        h = self.relu1_1(self.bn1_1(self.conv1_1(h)))
        h = self.relu1_2(self.bn1_2(self.conv1_2(h)))

        h = self.relu2_1(self.bn2_1(self.conv2_1(h)))
        h = self.relu2_2(self.bn2_2(self.conv2_2(h)))

        h = self.relu3_1(self.bn3_1(self.conv3_1(h)))
        h = self.relu3_2(self.bn3_2(self.conv3_2(h)))
        out = self.conv3_3(h)

        h = self.relu4_1(self.bn4_1(self.conv4_1(out)))
        h = self.relu4_2(self.bn4_2(self.conv4_2(h)))
        h = self.relu4_3(self.bn4_3(self.conv4_3(h)))

        h = self.relu5_1(self.bn5_1(self.conv5_1(h)))
        h = self.relu5_2(self.bn5_2(self.conv5_2(h)))
        h = self.relu5_3(self.bn5_3(self.conv5_3(h)))
        #h = torch.sigmoid(self.fc1(h))
        h = self.fc1(h)
        return {
            'recon': h,
            'vessel': out,
        }

class FCN32sBranch(BaseModel):

    def __init__(self, inp_channels=3, out_channels=4):
        super(FCN32sBranch, self).__init__()
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        # conv1
        self.conv1_1 = nn.Conv2d(inp_channels, 64, 3, padding=1)
        self.relu1_1 = nn.LeakyReLU()
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.LeakyReLU()
        self.bn1_2 = nn.BatchNorm2d(64)

        # conv2
        self.conv2_1 = nn.Conv2d(64, 32, 3, padding=1)
        self.relu2_1 = nn.LeakyReLU()
        self.bn2_1 = nn.BatchNorm2d(32)
        self.conv2_2 = nn.Conv2d(32, 32, 3, padding=1)
        self.relu2_2 = nn.LeakyReLU()
        self.bn2_2 = nn.BatchNorm2d(32)

        # conv3
        self.conv3_1 = nn.Conv2d(32, 32, 5, padding=2)
        self.relu3_1 = nn.LeakyReLU()
        self.bn3_1 = nn.BatchNorm2d(32)
        self.conv3_2 = nn.Conv2d(32, 32, 5, padding=2)
        self.relu3_2 = nn.LeakyReLU()
        self.bn3_2 = nn.BatchNorm2d(32)

        self.conv3_3 = nn.Conv2d(32, self.out_channels, 1, padding=0)

        # conv4
        self.conv4_1 = nn.Conv2d(32, 32, 5, padding=2)
        self.relu4_1 = nn.LeakyReLU()
        self.bn4_1 = nn.BatchNorm2d(32)
        self.conv4_2 = nn.Conv2d(32, 32, 5, padding=2)
        self.relu4_2 = nn.LeakyReLU()
        self.bn4_2 = nn.BatchNorm2d(32)
        self.conv4_3 = nn.Conv2d(32, 32, 5, padding=2)
        self.relu4_3 = nn.LeakyReLU()
        self.bn4_3 = nn.BatchNorm2d(32)

        # conv5
        self.conv5_1 = nn.Conv2d(32, 32, 5, padding=2)
        self.relu5_1 = nn.LeakyReLU()
        self.bn5_1 = nn.BatchNorm2d(32)
        self.conv5_2 = nn.Conv2d(32, 32, 5, padding=2)
        self.relu5_2 = nn.LeakyReLU()
        self.bn5_2 = nn.BatchNorm2d(32)
        self.conv5_3 = nn.Conv2d(32, 32, 3, padding=1)
        self.relu5_3 = nn.LeakyReLU()
        self.bn5_3 = nn.BatchNorm2d(32)

        # fc (this is where we will get the vessel)
        self.fc1 = nn.Conv2d(32, inp_channels, 3, padding=1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if False and isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        h = x['image']
        h = self.relu1_1(self.bn1_1(self.conv1_1(h)))
        h = self.relu1_2(self.bn1_2(self.conv1_2(h)))

        h = self.relu2_1(self.bn2_1(self.conv2_1(h)))
        h = self.relu2_2(self.bn2_2(self.conv2_2(h)))

        h = self.relu3_1(self.bn3_1(self.conv3_1(h)))
        h = self.relu3_2(self.bn3_2(self.conv3_2(h)))

        out = self.conv3_3(h)

        h = self.relu4_1(self.bn4_1(self.conv4_1(h)))
        h = self.relu4_2(self.bn4_2(self.conv4_2(h)))
        h = self.relu4_3(self.bn4_3(self.conv4_3(h)))

        h = self.relu5_1(self.bn5_1(self.conv5_1(h)))
        h = self.relu5_2(self.bn5_2(self.conv5_2(h)))
        h = self.relu5_3(self.bn5_3(self.conv5_3(h)))
        #h = torch.sigmoid(self.fc1(h))
        h = self.fc1(h)
        return {
            'recon': h,
            'vessel': out,
        }




class FCN32s(BaseModel):

    def __init__(self, inp_channels=3, out_channels=4):
        super(FCN32s, self).__init__()
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        # conv1
        self.conv1_1 = nn.Conv2d(inp_channels, 64, 3, padding=1)
        self.relu1_1 = nn.LeakyReLU()
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.LeakyReLU()
        self.bn1_2 = nn.BatchNorm2d(64)

        # conv2
        self.conv2_1 = nn.Conv2d(64, 32, 3, padding=1)
        self.relu2_1 = nn.LeakyReLU()
        self.bn2_1 = nn.BatchNorm2d(32)
        self.conv2_2 = nn.Conv2d(32, 32, 3, padding=1)
        self.relu2_2 = nn.LeakyReLU()
        self.bn2_2 = nn.BatchNorm2d(32)

        # conv3
        self.conv3_1 = nn.Conv2d(32, 32, 5, padding=2)
        self.relu3_1 = nn.LeakyReLU()
        self.bn3_1 = nn.BatchNorm2d(32)
        self.conv3_2 = nn.Conv2d(32, 32, 5, padding=2)
        self.relu3_2 = nn.LeakyReLU()
        self.bn3_2 = nn.BatchNorm2d(32)

        self.conv3_3 = nn.Conv2d(32, self.out_channels, 1, padding=0)

        # conv4
        self.conv4_1 = nn.Conv2d(self.out_channels, 32, 5, padding=2)
        self.relu4_1 = nn.LeakyReLU()
        self.bn4_1 = nn.BatchNorm2d(32)
        self.conv4_2 = nn.Conv2d(32, 32, 5, padding=2)
        self.relu4_2 = nn.LeakyReLU()
        self.bn4_2 = nn.BatchNorm2d(32)
        self.conv4_3 = nn.Conv2d(32, 32, 5, padding=2)
        self.relu4_3 = nn.LeakyReLU()
        self.bn4_3 = nn.BatchNorm2d(32)

        # conv5
        self.conv5_1 = nn.Conv2d(32, 32, 5, padding=2)
        self.relu5_1 = nn.LeakyReLU()
        self.bn5_1 = nn.BatchNorm2d(32)
        self.conv5_2 = nn.Conv2d(32, 32, 5, padding=2)
        self.relu5_2 = nn.LeakyReLU()
        self.bn5_2 = nn.BatchNorm2d(32)
        self.conv5_3 = nn.Conv2d(32, 32, 3, padding=1)
        self.relu5_3 = nn.LeakyReLU()
        self.bn5_3 = nn.BatchNorm2d(32)

        # fc (this is where we will get the vessel)
        self.fc1 = nn.Conv2d(32, inp_channels, 3, padding=1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if False and isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        h = x['image']
        h = self.relu1_1(self.bn1_1(self.conv1_1(h)))
        h = self.relu1_2(self.bn1_2(self.conv1_2(h)))

        h = self.relu2_1(self.bn2_1(self.conv2_1(h)))
        h = self.relu2_2(self.bn2_2(self.conv2_2(h)))

        h = self.relu3_1(self.bn3_1(self.conv3_1(h)))
        h = self.relu3_2(self.bn3_2(self.conv3_2(h)))
        out = self.conv3_3(h)

        h = self.relu4_1(self.bn4_1(self.conv4_1(out)))
        h = self.relu4_2(self.bn4_2(self.conv4_2(h)))
        h = self.relu4_3(self.bn4_3(self.conv4_3(h)))

        h = self.relu5_1(self.bn5_1(self.conv5_1(h)))
        h = self.relu5_2(self.bn5_2(self.conv5_2(h)))
        h = self.relu5_3(self.bn5_3(self.conv5_3(h)))
        #h = torch.sigmoid(self.fc1(h))
        h = self.fc1(h)
        return {
            'recon': h,
            'vessel': out,
        }


