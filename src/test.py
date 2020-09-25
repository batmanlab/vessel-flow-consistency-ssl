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


def smooth(ves, s=0.7):
    # image = [B, C, H, W]
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
    if "STARE" in args.dataset:
        config['data_loader']['type'] = "STAREDataLoader"
        config['data_loader']['args']['data_dir'] = config['data_loader']['args']['data_dir'].replace('DRIVE', 'STARE')


    training = True if args.train else False
    trainstr = "train" if args.train != 0 else "test"

    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=512,
        shuffle=False,
        toy=config['data_loader']['args']['toy'],
        preprocessing=config['data_loader']['args'].get('preprocessing'),
        validation_split=0.0,
        training=training,
        augment=False,
        num_workers=2
    )

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
        vesselfunc = v1_sqmax_vesselness
    elif config['loss'] == 'vessel_loss_2dv1_sq':
        vesselfunc = v1_sq_vesselness
    else:
        assert False, 'Unknown loss function {}'.format(config['loss'])
    print(vesselfunc)

    ## Check with curved vesselness
    # vesselfunc = v2_curved_vesselness
    # parallel_scale = [10, 10]

    parallel_scale = config.config['loss_args'].get('parallel_scale', 2)
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
            #
            # save sample images, or do something with output here
            #
            vessel_type = config.get('vessel_type', 'light')
            mask = data.get('mask')
            if mask is not None:
                mask = mask.cpu()

            v2 = output['vessel'][:, 2:4].cpu()

            # Change this for different vesselness modes
            if True:
                ves = vesselfunc(data['image'].cpu(), v2, vtype=vessel_type, mask=mask, is_crosscorr=args.crosscorr, parallel_scale=parallel_scale)
                ves = ves.data.cpu().numpy()
                ves = smooth(ves)
            else:
                ves = vesselfunc(data['image'].cpu(), v2, vtype=vessel_type, mask=mask, is_crosscorr=False, parallel_scale=parallel_scale)
                ves = smooth(ves)
                ves = v2_avg(ves, v2, vtype='light', mask=mask, is_crosscorr=False, parallel_scale=parallel_scale)
                ves = ves.data.cpu().numpy()


            # Add the other frangi-like term
            '''
            ves2 = v2transpose_vesselness(data['image'].cpu(), output['vessel'][:, 2:4].cpu(), vtype=vessel_type, mask=mask, is_crosscorr=False)
            ves2 = ves2.data.cpu().numpy()
            ves = ves*np.exp(-ves2)
            '''

            # computing loss, metrics on test set
            with open('{}_vesselness.pkl'.format(trainstr), 'wb') as fi:
                pkl.dump(ves, fi)

            # store everything in another pickle file
            with open('{}_analysis.pkl'.format(trainstr), 'wb') as fi:
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
    args.add_argument('--crosscorr', default=1, type=int)
    args.add_argument('--dataset', default="", type=str)
    args.add_argument('--train', default=0, type=int)

    config = ConfigParser.from_args(args)
    args = args.parse_args()
    main(config, args)
