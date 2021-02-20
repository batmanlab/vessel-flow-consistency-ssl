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
        batch_size=args.batch_size,
        shuffle=False,
        validation_split=0.0,
        training=training,
        num_workers=2,
    )

    ## Vesselness function (3d versions only here)
    ## TODO
    if config.config['loss'] == 'vessel_loss_2d_sq':
        vesselfunc = v2_sq_vesselness
    elif config.config['loss'] == 'vessel_loss_3d':
        vesselfunc = v13d_sq_vesselness_test
        #vesselfunc = v13d_sq_vesselness
    elif config.config['loss'] == 'vessel_loss_3d_bifurc':
        vesselfunc = v13d_sq_jointvesselness
    elif config.config['loss'] == 'vessel_loss_3dmax':
        vesselfunc = v13d_sqmax_vesselness
    else:
        assert False, 'Unknown loss function {}'.format(config['loss'])
    print(vesselfunc)
    input("Enter to continue. ")

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

    print("Dataset has size: {}".format(len(data_loader.dataset)))

    with torch.no_grad():
        # Store all stuff here
        vesselness = dict(imgid=None, img=None, count=None)

        for j, data in enumerate(tqdm(data_loader)):
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
            #print(ves.shape)

            # From this vesselness, use it to add to current image
            B = ves.shape[0]
            for i in tqdm(range(B)):
                imgid = data['imgid'][i]
                shape = list(data['shape'][i])

                # New image, save old image and create new templates
                if imgid != vesselness['imgid']:

                    # Save it first
                    v_id = vesselness['imgid']
                    v_img = vesselness['img']

                    # Save this image if it exists
                    if v_id is not None:
                        v_cnt = np.maximum(1, vesselness['count'])

                        ves_img = v_img / v_cnt
                        zerocount = np.where(v_cnt == 0)
                        if zerocount[0] != []:   # Print a warning is some location has zero index
                            print("{} index has zero count in location {}".format(v_id, zerocount))

                        outputfile = 'VESSEL12output/ours_{}.npy'.format(v_id)
                        print(v_cnt.min(), v_cnt.max())
                        with open(outputfile, 'wb') as fi:
                            np.save(fi, ves_img)
                        print("Saved to {}".format(outputfile))

                    # Create new images
                    vesselness['imgid'] = imgid
                    vesselness['img'] = np.zeros(shape)
                    vesselness['count'] = np.zeros(shape)

                # Save patch to new image
                #img = data['image'][i, 0].cpu().data.numpy()   # [P, P, P]
                img = ves[i, 0]
                _, H, W, D = data['startcoord'][i].cpu().data.numpy()
                pH, pW, pD = img.shape

                # Add to patch
                vesselness['img'][H:H+pH, W:W+pW, D:D+pD] = img + vesselness['img'][H:H+pH, W:W+pW, D:D+pD]
                vesselness['count'][H:H+pH, W:W+pW, D:D+pD] = 1 + vesselness['count'][H:H+pH, W:W+pW, D:D+pD]

                #print(vesselness['img'].min(), vesselness['img'].max(), vesselness['img'].mean(), )
                #print(vesselness['count'].min(), vesselness['count'].max(), vesselness['count'].mean(), )


        # The last image wont be saved, save it here
        v_id = vesselness['imgid']
        v_img = vesselness['img']
        v_cnt = np.maximum(1, vesselness['count'])

        ves_img = v_img / v_cnt
        zerocount = np.where(v_cnt == 0)
        if zerocount[0] != []:   # Print a warning is some location has zero index
            print("{} index has zero count in location {}".format(v_id, zerocount))

        outputfile = 'VESSEL12output/ours_{}.npy'.format(v_id)
        print(v_cnt.min(), v_cnt.max())
        with open(outputfile, 'wb') as fi:
            np.save(fi, ves_img)
        print("Saved to {}".format(outputfile))



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
    args.add_argument('--patientIDs', default=None, type=str, nargs='+')
    args.add_argument('--batch_size', default=512, type=int,)

    config = ConfigParser.from_args(args)
    args = args.parse_args()
    main(config, args)
