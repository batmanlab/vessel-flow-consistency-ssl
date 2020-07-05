from torchvision import datasets, transforms
from base import BaseDataLoader
from data_loader.datasets import *


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class DRIVEDataLoader(BaseDataLoader):
    '''
    Dataloader for DRIVE dataset
    '''
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, toy=False):
        self.data_dir = data_dir
        self.dataset = DriveDataset(self.data_dir, train=training, toy=toy)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class ToyStrLineLoader(BaseDataLoader):
    '''
    Dataloader for straight lines
    '''
    def __init__(self, img_size, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True ):
        self.img_size = img_size
        self.dataset = ToyStrLines(self.img_size, train=training)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class ToySlantLineLoader(BaseDataLoader):
    '''
    Dataloader for lines with changing thickness
    '''
    def __init__(self, img_size, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True ):
        self.img_size = img_size
        self.dataset = ToySlantLines(self.img_size, train=training)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class ToyCurvedLineLoader(BaseDataLoader):
    '''
    Dataloader for curved lines
    '''
    def __init__(self, img_size, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.img_size = img_size
        self.dataset = ToyCurvedLines(self.img_size, train=training)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
