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
from scipy.ndimage import gaussian_filter
from parse_config import ConfigParser
from utils import dir2flow_2d, v2vesselness, v2transpose_vesselness, overlay, overlay_quiver
from utils.util import *
from model.loss import v2_avg

def to_device(data, device):
    for k, v in data.items():
        data[k] = v.to(device)
    return data


def smooth(ves, s=1):
    # image = [B, C, H, W]
    smoothves = ves * 0
    B, C, H, W, D = ves.shape
    for b in range(B):
        for c in range(C):
            if isinstance(ves, torch.Tensor):
                sm = gaussian_filter(ves[b, c].data.numpy(), sigma=s)
                smoothves[b, c] = torch.Tensor(sm)
            else:
                smoothves[b, c] = gaussian_filter(ves[b, c], sigma=s)
    return smoothves


def main(config, args):
    logger = config.get_logger('test')

    training = True if args.train else False
    trainstr = "train" if args.train != 0 else "test"

    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=1,
        shuffle=False,
        validation_split=0.0,
        training=training,
        patientIDs=config['data_loader']['args'].get('patientIDs'),
        sigma=config['data_loader']['args'].get('sigma'),
        num_workers=2,
    )

    ## Vesselness function (3d versions only here)
    ## TODO
    if config.config['loss'] == 'vessel_loss_2d_sq':
        vesselfunc = v2_sq_vesselness
    elif config.config['loss'] == 'vessel_loss_3d':
        vesselfunc = v13d_sq_vesselness_test
    else:
        assert False, 'Unknown loss function {}'.format(config['loss'])
    print(vesselfunc)

    if 'max' in config['loss']:
        s = 1
    else:
        s = 0.7
    print("Smoothness {}".format(s))

    # More information
    parallel_scale = config.config['loss_args'].get('parallel_scale', 2)
    sv_range = config.config['loss_args'].get('sv_range')

    # build model architecture
    model = config.init_obj('arch', module_arch)
    model.eval()
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)

    # get nsamples
    nsample = config['loss_args'].get('num_samples_template', 12)

    # Load model
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
        # Store all stuff here
        vesselness = []
        alloutputs = []

        for i, data in enumerate(tqdm(data_loader)):
            ## Get output from model
            data = to_device(data, device)
            output = model(data)
            ## Get vessels
            vessel_type = config.get('vessel_type', 'light')
            mask = data.get('mask')
            if mask is not None:
                mask = mask.cpu()


            ## Change this for different vesselness modes
            ves = vesselfunc(data['image'], output, nsample=nsample, vtype=vessel_type, mask=mask, is_crosscorr=args.crosscorr, parallel_scale=parallel_scale, sv_range=sv_range)
            ves = ves.data.cpu().numpy()
            ves = smooth(ves, s)

            ## Put all outputs to cpu
            for k, v in output.items():
                try:
                    output[k] = v.cpu()
                except:
                    pass

            vesselness.append(ves)
            alloutputs.append(output)


        # computing loss, metrics on test set
        with open('{}_vesselness_3d.pkl'.format(trainstr), 'wb') as fi:
            pkl.dump(vesselness, fi)

        # store everything in another pickle file
        with open('{}_analysis_3d.pkl'.format(trainstr), 'wb') as fi:
            torch.save(alloutputs, fi)



if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('--run_id', default='test')
    args.add_argument('--crosscorr', default=0, type=int)
    args.add_argument('--dataset', default="", type=str)
    args.add_argument('--train', default=0, type=int)

    config = ConfigParser.from_args(args)
    args = args.parse_args()
    main(config, args)
