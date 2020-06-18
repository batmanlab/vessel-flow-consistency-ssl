import torch

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

# Metrics for vessel segmentation
def recon_error_l2(output, data):
    recon = output['recon']
    vessel = output['vessel']
    image = data['image']
    return torch.mean((image - recon)**2)

def recon_error_ce(output, data, eps=1e-10):
    recon = output['recon']
    vessel = output['vessel']
    image = data['image']
    ce = -image * torch.log(eps + recon) - (1 - image) * torch.log(eps + 1 - recon)
    return torch.mean(ce)

