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

'''
#print(a.shape)
x = net.conv1(a)
x = net.bn1(x)
x = net.relu(x)
#print(x.shape)

x = net.layer1(x)
#print(x.shape)
x = net.layer2(x)
#print(x.shape)
x = net.layer3(x)
#print(x.shape)
x = net.layer4(x)
#print(x.shape)

#print("Upsampling now")
x = net.uplayer4(x)
#print(x.shape)
x = net.uplayer3(x)
#print(x.shape)
x = net.uplayer2(x)
#print(x.shape)
x = net.uplayer1(x)
#print(x.shape)

x = net.finalconv(x)
#print(x.shape)
'''
