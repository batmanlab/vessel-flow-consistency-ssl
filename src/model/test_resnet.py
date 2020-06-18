import torch
import resnet

inp = 3
out = 7

net = resnet.resnet18(inp, out)
a = torch.rand(2, inp, 256, 256)
print(a.shape)
x = net.conv1(a)
x = net.bn1(x)
x = net.relu(x)
print(x.shape)

x = net.layer1(x)
print(x.shape)
x = net.layer2(x)
print(x.shape)
x = net.layer3(x)
print(x.shape)
x = net.layer4(x)
print(x.shape)

print("Upsampling now")
x = net.uplayer4(x)
print(x.shape)
x = net.uplayer3(x)
print(x.shape)
x = net.uplayer2(x)
print(x.shape)
x = net.uplayer1(x)
print(x.shape)

x = net.finalconv(x)
print(x.shape)
