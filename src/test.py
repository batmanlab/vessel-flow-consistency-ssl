''' 
Test script, loads the trained model, loads the test data loader and saves the output vesselness images
This is for all 2D datasets (DRIVE, STARE, HRF, RITE).
'''
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
    # Gaussian smooth the image (mostly used to smooth out the vessel fields, etc.)
    # image = [B, C, H, W]
    # Iterate over the batch and channel dimensions and smooth out each image
    smoothves = ves * 0
    B, C, H, W = ves.shape
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

    # setup data_loader instances
    # Special case for STARE dataset since transfer learning experiments needed the trained model from DRIVE
    # (given by its configuration file), 
    # but the evaluation needed to be done on the STARE dataset
    if "STARE" in args.dataset:
        config['data_loader']['type'] = "STAREDataLoader"
        config['data_loader']['args']['data_dir'] = config['data_loader']['args']['data_dir'].replace('DRIVE', 'STARE')

    # For training(val) or test dataset
    training = True if args.train else False

    # data loader 
    # get data directory from the config, a large batch size for inference,
    # and preprocessing mode
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=512,
        shuffle=False,
        toy=config['data_loader']['args'].get('toy'),
        preprocessing=config['data_loader']['args'].get('preprocessing'),
        validation_split=0.0,
        training=training,
        augment=False,
        num_workers=2
    )
    print(len(data_loader.dataset))

    ## Vesselness function
    if config.config['loss'] == 'vessel_loss_2d_sq':
        vesselfunc = v2_sq_vesselness
    elif config.config['loss'] == 'vessel_loss_2d_path':
        vesselfunc = v2_path_vesselness
    elif config['loss'] == 'vessel_loss_2d_dampen':
        vesselfunc = v2vesselness
    elif config['loss'] == 'vessel_loss_2d_curved':
        vesselfunc = v2_curved_vesselness
    elif config['loss'] == 'vessel_loss_2d_sqmax':
        vesselfunc = v2_sqmax_vesselness
    elif config['loss'] == 'vessel_loss_2dv1_sqmax':
        vesselfunc = v1_sqmax_vesselness_test
    elif config['loss'] == 'vessel_loss_2dv1_sq':
        vesselfunc = v1_sq_vesselness_test
    elif config['loss'] == 'vessel_loss_2dv1_bifurcmax':
        vesselfunc = v1_sqmax_jointvesselness_test
    elif config['loss'] == 'vessel_loss_2dv1_bifurconlymax':
        vesselfunc = v1_sqmax_bifurconly_test
    else:
        assert False, 'Unknown loss function {}'.format(config['loss'])
    print(vesselfunc)

    # Different smoothness for max and non-max filters
    if 'max' in config['loss']:
        s = 1
    else:
        s = 0.7
    print("Smoothness {}".format(s))

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
            vessel_type = config.get('vessel_type', 'light')
            mask = data.get('mask')

            # Change this for different vesselness modes
            ves = vesselfunc(data['image'], output, vtype=vessel_type, mask=mask, is_crosscorr=args.crosscorr, parallel_scale=parallel_scale, sv_range=sv_range)
            ves = ves.data.cpu().numpy()
            ves = smooth(ves, s)

            # Move outputs to CPU
            for k, v in output.items():
                try:
                    output[k] = v.cpu()
                except:
                    pass

            # computing loss, metrics on test set
            vfilename = input("Enter filename for vesselness. ")
            with open(vfilename, 'wb') as fi:
                pkl.dump(ves, fi)

            # store other items like radius, vessel flow, etc. in another pickle file 
            # for debugging and/or analysis
            analysisfilename = input("Enter filename to save analysis. ")
            with open(analysisfilename, 'wb') as fi:
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
    args.add_argument('--crosscorr', default=0, type=int)
    args.add_argument('--dataset', default="", type=str)
    args.add_argument('--train', default=0, type=int)

    config = ConfigParser.from_args(args)
    args = args.parse_args()
    main(config, args)
