import json
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import torch
from numpy import pi
from skimage.color import hsv2rgb

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
