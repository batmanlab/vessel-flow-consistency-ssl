import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from utils import dir2flow_2d, v2vesselness, overlay, overlay_quiver
from utils.util import *

class VesselTrainer(BaseTrainer):
    """
    2D Vessel Trainer class

    This trainer class is valid for all 2D datasets.

    model: network architecture
    criterion: loss function to train on
    metric_ftns: set of metrics to check for validation
    optimizer: optimizer for the model
    config: json file for config
    data_loader: training loader
    valid_data_loader: validation training loader
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
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
        ''' vesselness function zoo

        The vesselness function converts the network outputs (radius, vessel flow, bifurcation flow) into a 'vesselness' score.
        This vesselness score is then used to train the template loss in the self-supervised flow consistency framework.

        Refer to the definitions of the individual functions for their implementation.
        Some 'vesselness' functions are simply proxies that return the network output as the vesselness.
        This output can then be trained with whatever self-supervised loss function we want.
        '''
        if self.config['loss'] == 'vessel_loss_2d_sq':
            self.vesselfunc = v2_sq_vesselness
        elif self.config['loss'] == 'vessel_loss_2d_path':
            self.vesselfunc = v2_path_vesselness
        elif self.config['loss'] == 'vessel_loss_2d_dampen':
            self.vesselfunc = v2vesselness
        elif self.config['loss'] == 'vessel_loss_2d_curved':
            self.vesselfunc = v2_curved_vesselness
        elif self.config['loss'] == 'vessel_loss_2d_ode':
            self.vesselfunc = v2_ode_vesselness
        elif self.config['loss'] == 'vessel_loss_2d_sqmax':
            self.vesselfunc = v2_sqmax_vesselness
        elif self.config['loss'] == 'vessel_loss_2dv1_sqmax':
            self.vesselfunc = v1_sqmax_vesselness
        elif self.config['loss'] == 'vessel_loss_2dv1_sq':
            self.vesselfunc = v1_sq_vesselness
            # Bifurcation variants incoming
        elif self.config['loss'] == 'vessel_loss_2dv1_bifurcmax':
            self.vesselfunc = v1_sqmax_jointvesselness
        elif self.config['loss'] == 'vessel_loss_2dv1_bifurconlymax':
            self.vesselfunc = v1_sqmax_bifurconly_test
        elif self.config['loss'] == 'self_supervised_loss':
            self.vesselfunc = self_supervised_image
        elif self.config['loss'] == 'colorization_loss':
            self.vesselfunc = self_supervised_image
        elif self.config['loss'] == 'dice_loss':
            self.vesselfunc = self_supervised_image
        else:
            assert False, 'Unknown loss function {}'.format(self.config['loss'])
        print("Using vesselness function", self.vesselfunc)


    def _to_device(self, data):
        '''Simple helper function to move all data (given as a key,value dict) values into the given device (which is usually a specified GPU)
        
        Currently only supports a single GPU.
        '''
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
        normflow = params.get('normalize_flow', True)
        normflowrev = params.get('normalize_flow_rev', True)
        quiverscale = params.get('quiver_scale', 2)

        self.model.train()
        self.train_metrics.reset()
        for batch_idx, data in enumerate(self.data_loader):
            # Move tensors to device
            data = self._to_device(data)
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

            # Create outputviz to have detached values
            outputviz = {}
            for k, v in output.items():
                try:
                    outputviz[k] = v.detach().cpu()
                except:
                    outputviz[k] = v

            # Every few steps, add some images
            if epoch % self.img_log_step == 0 and batch_idx == 0:
                self.writer.add_image('input', make_grid(0.5 + 0.5*data['image'].detach().cpu(), nrow=4, normalize=True))
                self.writer.add_image('recon', make_grid(0.5 + 0.5*output['recon'].detach().cpu(), nrow=4, normalize=True))
                self.writer.add_image('v_x', make_grid(output['vessel'][:, 0:1].detach().cpu(), nrow=4, normalize=True))
                self.writer.add_image('v_y', make_grid(output['vessel'][:, 1:2].detach().cpu(), nrow=4, normalize=True))
                self.writer.add_image('flow', make_grid(overlay_quiver(data['image'].detach().cpu(), output['vessel'][:, 0:2].detach().cpu(), quiverscale, normflow), nrow=4, normalize=True))
                self.writer.add_image('flow_rev', make_grid(overlay_quiver(data['image'].detach().cpu(), output['vessel'][:, 2:4].detach().cpu(), quiverscale, normflowrev), nrow=4, normalize=True))
                self.writer.add_image('v2_vesselness_only', make_grid(self.vesselfunc(data['image'].detach().cpu(), outputviz, vtype=vessel_type, \
                        mask = mask, v1 = output['vessel'][:, :2].detach().cpu(), parallel_scale=parallel_scale), nrow=4, normalize=True))
                ves = self.vesselfunc(data['image'].detach().cpu(), outputviz, vtype=vessel_type, mask=mask, v1 = output['vessel'][:, :2].detach().cpu(), parallel_scale=parallel_scale)
                overlay_img = overlay(data['image'].detach().cpu(), ves.data.detach().cpu())
                self.writer.add_image('v2_vesselness_overlay', make_grid(overlay_img, nrow=4, normalize=True))

                # Cross correlation vesselness
                ves = self.vesselfunc(data['image'].detach().cpu(), outputviz, vtype=vessel_type, mask=mask, is_crosscorr=True, v1 = output['vessel'][:, :2].detach().cpu(), parallel_scale=parallel_scale)
                self.writer.add_image('v2_vesselness_crosscorr', make_grid(ves.data.detach().cpu(), nrow=4, normalize=True))

            if batch_idx == self.len_epoch:
                break
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
        normflow = params.get('normalize_flow', True)
        normflowrev = params.get('normalize_flow_rev', True)
        quiverscale = params.get('quiver_scale', 2)

        # Get vessel type here
        vessel_type = self.config.get('vessel_type', 'light')
        parallel_scale = self.config.config['loss_args'].get('parallel_scale', 2)

        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, data in enumerate(self.valid_data_loader):
                # Move tensors to device
                data = self._to_device(data)

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

                # Create outputviz to have detached values
                outputviz = {}
                for k, v in output.items():
                    try:
                        outputviz[k] = v.detach().cpu()
                    except:
                        outputviz[k] = v

                if epoch % self.img_log_step == 0 and batch_idx == 0:
                    self.writer.add_image('input', make_grid(0.5 + 0.5*data['image'].cpu(), nrow=4, normalize=True))
                    self.writer.add_image('recon', make_grid(0.5 + 0.5*output['recon'].cpu(), nrow=4, normalize=True))
                    self.writer.add_image('v_x', make_grid(output['vessel'][:, 0:1].cpu(), nrow=4, normalize=True))
                    self.writer.add_image('v_y', make_grid(output['vessel'][:, 1:2].cpu(), nrow=4, normalize=True))
                    self.writer.add_image('flow', make_grid(overlay_quiver(data['image'].cpu(), output['vessel'][:, 0:2].cpu(), quiverscale, normflow), nrow=4, normalize=True))
                    self.writer.add_image('flow_rev', make_grid(overlay_quiver(data['image'].cpu(), output['vessel'][:, 2:4].cpu(), quiverscale, normflowrev), nrow=4, normalize=True))
                    self.writer.add_image('v2_vesselness_only', make_grid(self.vesselfunc(data['image'].cpu(), outputviz, vtype=vessel_type, \
                            mask=mask, v1 = output['vessel'][:, :2].cpu(), parallel_scale=parallel_scale), nrow=4, normalize=True))
                    ves = self.vesselfunc(data['image'].cpu(), outputviz, vtype=vessel_type, mask=mask, v1 = output['vessel'][:, :2].cpu(), parallel_scale=parallel_scale)
                    overlay_img = overlay(data['image'].cpu(), ves)
                    self.writer.add_image('v2_vesselness_overlay', make_grid(overlay_img, nrow=4, normalize=True))

                    # Cross correlation vesselness
                    ves = self.vesselfunc(data['image'].cpu(), outputviz, vtype=vessel_type, mask=mask, is_crosscorr=True, v1 = output['vessel'][:, :2].cpu(), parallel_scale=parallel_scale)
                    self.writer.add_image('v2_vesselness_crosscorr', make_grid(ves, nrow=4, normalize=True))

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
