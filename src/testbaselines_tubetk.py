''' 
Test script, for other baselines like Frangi, Sato, Hessian, Meijering vesselness methods
for the TubeTK dataset. The saved files are similar to the ones from `test3d.py` and the files
can then be used for comparison using `get_all_metrics_<dataset>.py` script.
'''
import argparse
import numpy as np
import torch
import multiprocessing as mp
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
from skimage.filters import frangi, sato, meijering, hessian
from functools import partial
from time import time

VESSELFUNC = {
        'frangi': frangi,
        'sato': sato,
        'hessian': hessian,
        'meijering': meijering,
}


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
        full_data=False,
        num_workers=4,
    )

    # Keep track of times
    times = []

    # Vessel function with partial args filled in
    vfunc = partial(VESSELFUNC[args.vesselfunc], sigmas=np.linspace(1, 12, 6), black_ridges=False, mode='constant')

    # Create pool for parallel jobs
    print(mp.cpu_count())

    pool = mp.Pool(args.num_cores)

    print("Dataset has size: {}".format(len(data_loader.dataset)))
    with torch.no_grad():
        # Store all stuff here
        vesselness = dict(imgid=None, img=None, count=0)

        for j, data in enumerate(tqdm(data_loader)):
            ## Get output from model
            imgs = data['image'].data.numpy()
            fragimg = [x[0] for x in imgs]

            # Record time
            t1 = time()
            fragves = pool.map(vfunc, fragimg)
            #fragves = [vfunc(x) for x in fragimg]
            ves = np.array(fragves)
            t2 = time()
            times.append(t2 - t1)
            #print(t2 - t1)

            # From this vesselness, use it to add to current image
            B = ves.shape[0]
            for i in range(B):
                imgid = data['imgid'][i]
                shape = list(data['shape'][i])

                # New image, save old image and create new templates
                if imgid != vesselness['imgid']:
                    # Save it first
                    v_id = vesselness['imgid']
                    v_img = vesselness['img']
                    #v_cnt = np.maximum(1, vesselness['count'])
                    v_cnt = vesselness['count']

                    # Save this image if it exists
                    if v_id is not None:
                        ves_img = v_img / v_cnt
                        zerocount = np.where(v_cnt == 0)
                        if zerocount[0] != []:   # Print a warning is some location has zero index
                            print("{} index has zero count in location {}".format(v_id, zerocount))

                        outputfile = '/ocean/projects/asc170022p/rohit33/TubeTKoutput/{}_{}.npy'.format(args.vesselfunc, v_id)
                        print(v_cnt.min(), v_cnt.max())
                        with open(outputfile, 'wb') as fi:
                            np.save(fi, ves_img)
                        print("Saved to {}".format(outputfile))

                    # Create new images
                    vesselness['imgid'] = imgid
                    vesselness['img'] = np.zeros(shape)
                    vesselness['count'] = np.zeros(shape)

                # Save patch to new image
                #img = ves[i, 0].cpu().data.numpy()   # [P, P, P]
                img = ves[i] + 0
                _, H, W, D = data['startcoord'][i].cpu().data.numpy()
                pH, pW, pD = img.shape

                # Add to patch
                vesselness['img'][H:H+pH, W:W+pW, D:D+pD] = img + vesselness['img'][H:H+pH, W:W+pW, D:D+pD]
                vesselness['count'][H:H+pH, W:W+pW, D:D+pD] = 1 + vesselness['count'][H:H+pH, W:W+pW, D:D+pD]


        # The last image wont be saved, save it here
        v_id = vesselness['imgid']
        v_img = vesselness['img']
        #v_cnt = np.maximum(1, vesselness['count'])
        v_cnt = vesselness['count']

        ves_img = v_img / v_cnt
        zerocount = np.where(v_cnt == 0)
        if zerocount[0] != []:   # Print a warning is some location has zero index
            print("{} index has zero count in location {}".format(v_id, zerocount))

        outputfile = '/ocean/projects/asc170022p/rohit33/TubeTKoutput/{}_{}.npy'.format(args.vesselfunc, v_id)
        print(v_cnt.min(), v_cnt.max())
        with open(outputfile, 'wb') as fi:
            np.save(fi, ves_img)
        print("Saved to {}".format(outputfile))

        # Print time stats
        meantime = np.mean(times)
        stdtime = np.std(times)
        print("Time for {}: {} +- {} sec".format(args.vesselfunc, meantime, stdtime))



if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('--run_id', default='testbaseline')
    args.add_argument('--crosscorr', default=0, type=int)
    args.add_argument('--dataset', default="", type=str)
    args.add_argument('--train', default=0, type=int)
    args.add_argument('--batch_size', default=512, type=int,)
    args.add_argument('--num_cores', default=4, type=int)
    args.add_argument('--vesselfunc', type=str, required=True, choices=list(VESSELFUNC.keys()))

    config = ConfigParser.from_args(args)
    args = args.parse_args()
    main(config, args)
