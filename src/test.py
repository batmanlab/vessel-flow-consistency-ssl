import argparse
import numpy as np
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
from matplotlib import pyplot as plt
import pickle as pkl
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from utils import dir2flow_2d, v2vesselness, v2transpose_vesselness, overlay, overlay_quiver
from utils.util import *

def to_device(data, device):
    for k, v in data.items():
        data[k] = v.to(device)
    return data


def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=512,
        shuffle=False,
        toy=config['data_loader']['args']['toy'],
        preprocessing=config['data_loader']['args'].get('preprocessing'),
        validation_split=0.0,
        training=True,
        augment=False,
        num_workers=2
    )

    ## Vesselness function
    if config.config['loss'] == 'vessel_loss_2d_sq':
        vesselfunc = v2_sq_vesselness
    elif config.config['loss'] == 'vessel_loss_2d_path':
        vesselfunc = v2_path_vesselness
    else:
        vesselfunc = v2vesselness

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)

    try:
        model.load_state_dict(state_dict)
    except:
        model.module.load_state_dict(state_dict)

    print("Using vessel function: {}".format(vesselfunc))
    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader)):
            #data, target = data.to(device), target.to(device)
            data = to_device(data, device)
            output = model(data)
            #
            # save sample images, or do something with output here
            #
            vessel_type = config.get('vessel_type', 'light')
            mask = data.get('mask')
            if mask is not None:
                mask = mask.cpu()

            ves = vesselfunc(data['image'].cpu(), output['vessel'][:, 2:4].cpu(), vtype=vessel_type, mask=mask, is_crosscorr=True)
            ves = ves.data.cpu().numpy()

            # Add the other frangi-like term
            '''
            ves2 = v2transpose_vesselness(data['image'].cpu(), output['vessel'][:, 2:4].cpu(), vtype=vessel_type, mask=mask, is_crosscorr=False)
            ves2 = ves2.data.cpu().numpy()
            ves = ves*np.exp(-ves2)
            '''

            # computing loss, metrics on test set
            with open('vesselness.pkl', 'wb') as fi:
                pkl.dump(ves, fi)

            # store everything in another pickle file
            with open('analysis.pkl', 'wb') as fi:
                torch.save({'data': data, 'output': output}, fi)

            I = np.random.randint(20)
            plt.subplot(121)
            plt.imshow(data['image'].cpu()[I, 0])
            plt.subplot(122)
            plt.imshow(ves[I, 0])
            plt.savefig('label.png')
            break


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('--run_id', default='test')

    config = ConfigParser.from_args(args)
    main(config)
