import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from utils import dir2flow_2d, v2vesselness, overlay, overlay_quiver
from utils.util import *

def squeeze_batch(data):
    # Squeeze the first and second indices into the first index
    # works with data with data dicts
    for k, v in data.items():
        shape = list(v.shape)
        shape = [shape[0]*shape[1]] + shape[2:]
        v = v.reshape(shape)
        data[k] = v
    return data

class COPDTrainer(BaseTrainer):
    """
    COPDTrainer class

    Although it is named COPDTrainer class, it is valid for all 3D datasets used in this project.

    model: network architecture
    criterion: loss function to train on
    metric_ftns: set of metrics to check for validation
    optimizer: optimizer for the model
    config: json file for config
    data_loader: training loader
    valid_data_loader: validation training loader
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader, squeeze=True,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.squeeze = squeeze
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.img_log_step = config['trainer'].get('img_log_step', 1)

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

        # Change v2vesselness function here
        # TODO
        if self.config['loss'] == 'vessel_loss_3d':
            self.vesselfunc = v13d_sq_vesselness
        elif self.config['loss'] == 'vessel_loss_3d_bifurc':
            self.vesselfunc = v13d_sq_jointvesselness
        else:
            self.vesselfunc = v13d_sqmax_vesselness
        print("Using vesselness function", self.vesselfunc)


    def _to_device(self, data):
        for k, v in data.items():
            data[k] = v.to(self.device)
        return data


    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        # Get parameters for quiver
        params = self.config['trainer']

        self.model.train()
        self.train_metrics.reset()
        for batch_idx, data in enumerate(self.data_loader):
            # Move tensors to device
            data = self._to_device(data)
            if self.squeeze:
                data = squeeze_batch(data)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, data, self.config)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.detach().item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, data).detach())

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.detach().item()))

            # Get vessel type here
            vessel_type = self.config.get('vessel_type', 'light')
            parallel_scale = self.config.config['loss_args'].get('parallel_scale', 2)
            mask = data.get('mask')
            if mask is not None:
                mask = mask.cpu()
            
            # Code for adding new images is removed because we have 3D patches now, which are hard and 
            # expensive to visualize over Tensorboard 
            if batch_idx == self.len_epoch:
                break

        # log all training metrics
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        # Get quiver params
        params = self.config['trainer']

        # Get vessel type here
        vessel_type = self.config.get('vessel_type', 'light')
        parallel_scale = self.config.config['loss_args'].get('parallel_scale', 2)

        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, data in enumerate(self.valid_data_loader):
                # Move tensors to device
                data = self._to_device(data)
                if self.squeeze:
                    data = squeeze_batch(data)

                output = self.model(data)
                loss = self.criterion(output, data, self.config)
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, data).detach())

                # Get vessel mask
                mask = data.get('mask')
                if mask is not None:
                    mask = mask.cpu()

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()


    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
