import torch
import resnet

inp = 1
out = 4

net = resnet.resnet18(inp, out)
a = {
        'image': torch.rand(1, inp, 32, 32)
}

out = net(a)
print(out['recon'].shape, out['vessel'].shape, a['image'].shape)
