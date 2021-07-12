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
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, toy=False, augment=True, preprocessing=False, idx=None):
        self.data_dir = data_dir
        self.dataset = DriveDataset(self.data_dir, train=training, toy=toy, preprocessing=preprocessing, \
                                    augment=augment, idx=idx)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class DRIVESupDataloader(BaseDataLoader):
    '''
    Dataloader for DRIVE dataset
    '''
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, toy=False, augment=True, preprocessing=False, idx=None):
        self.data_dir = data_dir
        self.dataset = DriveSupDataset(self.data_dir, train=training, toy=toy, preprocessing=preprocessing, \
                                    augment=augment, idx=idx)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class DRIVEContextDataLoader(BaseDataLoader):
    '''
    Dataloader for DRIVE dataset (noise)
    '''
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, toy=False, augment=True, preprocessing=False, idx=None):
        self.data_dir = data_dir
        self.dataset = DriveContextDataset(self.data_dir, train=training, toy=toy, preprocessing=preprocessing, \
                                    augment=augment, idx=idx)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)



class DRIVEGenesisDataLoader(BaseDataLoader):
    '''
    Dataloader for DRIVE dataset (noise)
    '''
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, toy=False, augment=True, preprocessing=False, idx=None):
        self.data_dir = data_dir
        self.dataset = DriveGenesisDataset(self.data_dir, train=training, toy=toy, preprocessing=preprocessing, \
                                    augment=augment, idx=idx)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class DRIVENoiseDataLoader(BaseDataLoader):
    '''
    Dataloader for DRIVE dataset (noise)
    '''
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, toy=False, augment=True, preprocessing=False, idx=None):
        self.data_dir = data_dir
        self.dataset = DriveNoiseDataset(self.data_dir, train=training, toy=toy, preprocessing=preprocessing, \
                                    augment=augment, idx=idx)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class COPDDataLoader(BaseDataLoader):
    '''
    Dataloader for COPD dataset
    '''
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, augment=True, patientIDs=None, sigma=0):
        self.data_dir = data_dir
        self.dataset = COPDDataset(data_dir, train=training, patientIDs=patientIDs, minibatch=batch_size, augment=augment)
        super().__init__(self.dataset, 1, shuffle, validation_split, num_workers)


class VascuDataLoader(BaseDataLoader):
    '''
    Data loader for VascuSynth dataset
    '''
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, patientIDs=None, sigma=0.1):
        self.data_dir = data_dir
        self.dataset = VascuDataset(data_dir, train=training, sigma=sigma)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class STAREDataLoader(BaseDataLoader):
    '''
    Dataloader for DRIVE dataset
    '''
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, toy=False, augment=True, preprocessing=False, idx=None):
        self.data_dir = data_dir
        self.dataset = StareDataset(self.data_dir, train=training, toy=toy, preprocessing=preprocessing, \
                                    augment=augment, idx=idx)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class STARESupDataLoader(BaseDataLoader):
    '''
    Dataloader for DRIVE dataset
    '''
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, toy=False, augment=True, preprocessing=False, idx=None):
        self.data_dir = data_dir
        self.dataset = StareSupDataset(self.data_dir, train=training, toy=toy, preprocessing=preprocessing, \
                                    augment=augment, idx=idx)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class DRIVEContrastDataLoader(BaseDataLoader):
    '''
    Dataloader for DRIVE dataset
    '''
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, toy=False, augment=True, preprocessing=True, idx=None):
        self.data_dir = data_dir
        self.dataset = DriveContrastDataset(self.data_dir, train=training, toy=toy, preprocessing=preprocessing, augment=augment, idx=idx)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class DRIVEColorDataLoader(BaseDataLoader):
    '''
    Dataloader for DRIVE dataset
    '''
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, toy=False, augment=True, preprocessing=True, idx=None):
        self.data_dir = data_dir
        self.dataset = DriveColorDataset(self.data_dir, train=training, toy=toy, preprocessing=preprocessing, augment=augment, idx=idx)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class ToyStrLineLoader(BaseDataLoader):
    '''
    Dataloader for straight lines
    '''
    def __init__(self, img_size, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, holesize=0, noise=0, dark=False):
        self.img_size = img_size
        self.dataset = ToyStrLines(self.img_size, train=training, holesize=holesize, noise=noise, dark=dark)
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


class TubeTKMRADataLoader(BaseDataLoader):
    '''
    Data loader for VascuSynth dataset
    '''
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, full_data=True, offset=0):
        self.data_dir = data_dir
        if full_data:
            self.dataset = TubeTKFullDataset(data_dir, train=training, offset=offset)
        else:
            self.dataset = TubeTKDataset(data_dir, train=training, offset=offset)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class Vessel12DataLoader(BaseDataLoader):
    '''
    Data loader for VascuSynth dataset
    '''
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, pad=0):
        self.data_dir = data_dir
        self.dataset = Vessel12Dataset(data_dir, train=training, pad=pad)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class IrcadDataLoader(BaseDataLoader):
    '''
    Data loader for VascuSynth dataset
    '''
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, pad=0):
        self.data_dir = data_dir
        self.dataset = IrcadDataset(data_dir, train=training, pad=pad)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


#############################
# More retinopathy datasets
#############################

class HRFDataLoader(BaseDataLoader):
    '''
    Dataloader for HRF dataset
    '''
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, augment=True, preprocessing=True, toy=False):
        self.data_dir = data_dir
        self.dataset = HRFDataset(self.data_dir, train=training, augment=augment)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class RITEDataLoader(BaseDataLoader):
    '''
    Dataloader for HRF dataset
    '''
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, augment=True, preprocessing=True, toy=False):
        self.data_dir = data_dir
        self.dataset = RITEDataset(self.data_dir, train=training, augment=augment)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

