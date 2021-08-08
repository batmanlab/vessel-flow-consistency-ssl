''' DEPRECATED: Use `loss.py` instead of this file. Most of loss functions use `v2` vesselness 
instead of `v1` vesselness.
'''
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .loss import resample_from_flow_2d, LOSS_FNs, flow_consistency_2d, L2, L1, v1_sq_vesselness

def v1_sqmax_vesselness(image, ves, nsample=12, vtype='light', mask=None, percentile=100, is_crosscorr=False, v1 = None, parallel_scale=2):
    ''' In this version, `v1` is the vessel direction, and `v2` is the perpendicular to the vessel direction
    '''
    response1 = 0.0
    response2 = 0.0
    i_range1 = []
    i_range2 = []

    v2 = ves*0
    v2[:, 1] = ves[:, 0]
    v2[:, 0] = -ves[:, 1]

    v1 = ves
    D = 0

    ## Collect the response from both sides into `half-vessels`
    # Then take the minimum response from both sides
    for sv in np.linspace(-parallel_scale, parallel_scale, nsample):
        # Get perp profile
        for s in np.linspace(-2, 2, nsample):
            filt = 2*int(abs(s) < 1) - 1
            D += filt
            i_val = resample_from_flow_2d(image, sv*v1 + s*v2)
            if is_crosscorr:
                if s < 0:
                    i_range1.append(i_val.detach()[:, None])
                else:
                    i_range2.append(i_val.detach()[:, None])
            # Compute the convolution I * f
            if s < 0:
                response1 = response1 + (i_val * filt)
            else:
                response2 = response2 + (i_val * filt)

    if is_crosscorr:
        # Take min of both sides
        i_std1 = torch.cat(i_range1, 1)
        i_std1 = i_std1.std(1, unbiased=False) + 1e-5
        response1 = response1 / i_std1

        i_std2 = torch.cat(i_range2, 1)
        i_std2 = i_std2.std(1, unbiased=False) + 1e-5
        response2 = response2 / i_std2

    # Correct the response accordingly
    if vtype == 'light':
        response = torch.min(response1, response2)
        pass
    elif vtype == 'dark':
        response = torch.min(-response1, -response2)
        #response = -response
    elif vtype == 'both':
        response = torch.min(torch.abs(response1), torch.abs(response2))
    else:
        raise NotImplementedError('{} type not supported in vesseltype'.format(vtype))

    # We got the response, now subtract from mean and multiply with optional mask
    response = response - response.min().detach()
    if mask is not None:
        response = response * mask
    return response



def vessel_loss_2dv1_sqmax(output, data, config, maxfilter=True):
    '''
    Master loss function of vessel self supervised learning
    '''
    args = config['loss_args']
    # Get all parameters
    num_dir = args['num_directions'] # Determine the directions
    unc = args['uncertainty'] # whether to use kappa uncertainty
    eps = args['eps']
    L_loss = LOSS_FNs[args['loss_intensity']]
    # Weights for different parts of total loss
    l_intensity = args.get('lambda_intensity')
    l_consistency = args.get('lambda_consistency')
    l_cosine = args.get('lambda_cosine')
    l_decoder = args.get('lambda_decoder')
    l_length = args.get('lambda_length')

    # Add extra parameters -> length and matching of template profile
    l_template = args.get('lambda_template')
    num_samples_template = args.get('num_samples_template', 12)
    l_perlength = args.get('lambda_perlength')
    detach = args.get('detach', True)
    # This parameter is for type of vessel
    vessel_type = config.get('vessel_type', 'light') # should be light, dark or both

    parallel_scale = args.get('parallel_scale', 2)

    # Check for max filter
    if maxfilter==False:
        vesselnessfun = v1_sq_vesselness
    else:
        vesselnessfun = v1_sqmax_vesselness

    # check if we want to minimize cross correlation
    is_crosscorr = args.get('is_crosscorr', False)

    # Get outputs and inputs
    recon = output['recon']
    vessel = output['vessel']
    image = data['image']

    # Take mask for all losses
    mask = data.get('mask', 1)
    if not args.get('use_mask', False):
        mask = 1

    # Now use the losses given in the config
    if num_dir == 2 and not unc:
        assert config['arch']['args']['out_channels'] == 4, 'Model with 2 directions and no uncertainty'
        # parameters are v1, v2
        loss = 0.0
        v1 = vessel[:, 2:]
        # Consistency loss
        if l_consistency:
            # Align v1 = v(x1) and v2 = v(x1 + v1) to be along same directions
            v1x = resample_from_flow_2d(v1, v1)
            loss = loss - l_consistency * F.cosine_similarity(v1, v1x).mean()

        # Check for cosine similarity
        if l_cosine:
            v1x = resample_from_flow_2d(v1, v1)
            sim = F.cosine_similarity(v1, v1x)
            loss = loss - l_consistency * torch.min(sim, sim*0).mean()

        # Check for decoder
        if l_decoder:
            loss = loss + l_decoder * L2(image, recon, mask=1)

        # Check profile by taking convolution with the template [-1 -1 1 1 1 1 -1 -1]
        vessel_conv = 0.0
        if l_template:
            vessel_conv = vesselnessfun(image, v1, num_samples_template, vessel_type, is_crosscorr=is_crosscorr, parallel_scale=parallel_scale)
            loss = loss + l_template * (1 - (mask*vessel_conv).mean())

        # Vessel intensity consistency loss
        if l_intensity:
            for scale in np.linspace(-parallel_scale/2., parallel_scale/2., 5):
                # Check for both directions for same intensity -> this will help in centerline prediction
                i_parent = resample_from_flow_2d(vessel_conv, scale*v1)
                loss = loss + l_intensity * (L_loss(image, i_parent, mask=mask))/5.0

    else:
        raise NotImplementedError

    # Return loss
    return loss
