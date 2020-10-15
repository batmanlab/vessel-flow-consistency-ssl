import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torchdiffeq import odeint_adjoint as odeint
from model.copdloss import *

def null_fn(*args):
    return None

#print_fn = print
print_fn = null_fn

# Output vesselness
def v2vesselness(image, ves, nsample=20, vtype='light', mask=None, percentile=100, is_crosscorr=False, v1=None, parallel_scale=2):
    response = 0.0
    i_range = []
    for s in np.linspace(-2, 2, nsample):
        filt = 2*int(abs(s) < 1) - 1
        i_val = resample_from_flow_2d(image, s*ves)
        if is_crosscorr:
            i_range.append(i_val.detach()[:, None])
        # Compute the convolution I * f
        response = response + (i_val * filt)/nsample

    # Normalize
    if is_crosscorr:
        i_range = torch.cat(i_range, 1)   # [B, 20, 1, H, W]
        i_std = i_range.std(1) + 1e-2    # [B, 1, H, W]
        response = response / i_std

    # Correct the response accordingly
    if vtype == 'light':
        pass
    elif vtype == 'dark':
        response = -response
    elif vtype == 'both':
        response = torch.abs(response)
    else:
        raise NotImplementedError('{} type not supported in vesseltype'.format(vtype))

    # We got the response, now subtract from mean and multiply with optional mask
    response = response - response.min().detach()
    if mask is not None:
        response = response * mask
    return response

def v2_avg(image, ves, nsample=12, vtype='light', mask=None, percentile=100, is_crosscorr=False, v1 = None, parallel_scale=2):
    response1 = 0.0
    response2 = 0.0
    i_range1 = []
    i_range2 = []

    v1 = ves*0
    v1[:, 1] = ves[:, 0]
    v1[:, 0] = -ves[:, 1]

    D = 0
    for sv in np.linspace(-parallel_scale, parallel_scale, nsample):
        # Get perp profile
        for s in np.linspace(-1, 1, nsample):
            filt = 1
            D += filt
            i_val = resample_from_flow_2d(image, s*ves + sv*v1)
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

    response = response1 + response2
    # Normalize
    # Correct the response accordingly
    if vtype == 'light':
        pass
    elif vtype == 'dark':
        response = -response
    elif vtype == 'both':
        response = 2*torch.abs(response)
    else:
        raise NotImplementedError('{} type not supported in vesseltype'.format(vtype))

    # We got the response, now subtract from mean and multiply with optional mask
    response = response - response.min().detach()
    if mask is not None:
        response = response * mask
    return response


def v2_sqmax_vesselness(image, ves, nsample=12, vtype='light', mask=None, percentile=100, is_crosscorr=False, v1 = None, parallel_scale=2):
    response1 = 0.0
    response2 = 0.0
    i_range1 = []
    i_range2 = []

    v1 = ves*0
    v1[:, 1] = ves[:, 0]
    v1[:, 0] = -ves[:, 1]

    D = 0
    for sv in np.linspace(-parallel_scale, parallel_scale, nsample):
        # Get perp profile
        for s in np.linspace(-2, 2, nsample):
            filt = 2*int(abs(s) < 1) - 1
            D += filt
            i_val = resample_from_flow_2d(image, s*ves + sv*v1)
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

    # Uncomment below if using both sides
    #response = response1 + response2

    # Normalize
    if is_crosscorr:
        '''
        # Average of both sides (normal)
        i_range = torch.cat(i_range1 + i_range2, 1)
        i_std = i_range.std(1, unbiased=False) + 1e-5
        response = response / i_std / (nsample**2)
        # Check assertion
        idx = (-1 <= response)*(response <= 1)
        idx = (~idx).sum().item()
        assert idx == 0, '{} {}'.format(idx, np.prod(list(response.shape)))
        '''
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
        #response = 2*torch.abs(response)
    else:
        raise NotImplementedError('{} type not supported in vesseltype'.format(vtype))

    # We got the response, now subtract from mean and multiply with optional mask
    response = response - response.min().detach()
    if mask is not None:
        response = response * mask
    return response


# Output vesselness over square filters
def v2_sq_vesselness(image, ves, nsample=12, vtype='light', mask=None, percentile=100, is_crosscorr=False, v1 = None, parallel_scale=2):
    response1 = 0.0
    response2 = 0.0
    i_range1 = []
    i_range2 = []

    v1 = ves*0
    v1[:, 1] = ves[:, 0]
    v1[:, 0] = -ves[:, 1]

    D = 0
    for sv in np.linspace(-parallel_scale, parallel_scale, nsample):
        # Get perp profile
        for s in np.linspace(-2, 2, nsample):
            filt = 2*int(abs(s) < 1) - 1
            D += filt
            i_val = resample_from_flow_2d(image, s*ves + sv*v1)
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

    response = response1 + response2

    # Normalize
    if is_crosscorr:
        # Average of both sides (normal)
        i_range = torch.cat(i_range1 + i_range2, 1)
        i_std = i_range.std(1, unbiased=False) + 1e-5
        response = response / i_std / (nsample**2)
        # Check assertion
        idx = (-1 <= response)*(response <= 1)
        idx = (~idx).sum().item()
        assert idx == 0, '{} {}'.format(idx, np.prod(list(response.shape)))

    # Correct the response accordingly
    if vtype == 'light':
        pass
    elif vtype == 'dark':
        response = -response
    elif vtype == 'both':
        response = 2*torch.abs(response)
    else:
        raise NotImplementedError('{} type not supported in vesseltype'.format(vtype))

    # We got the response, now subtract from mean and multiply with optional mask
    response = response - response.min().detach()
    if mask is not None:
        response = response * mask
    return response


def nll_loss(output, target):
    return F.nll_loss(output, target)

def L1(i1, i2=0, mask=1):
    return (torch.abs(i1 - i2)*mask).mean()

def L2(i1, i2=0, mask=1):
    return torch.mean(mask * (i1 - i2)**2)

def CE(i1, i2, eps=1e-10):
    out = -i1*torch.log(i2 + eps) - (1 - i1)*torch.log(1 - i2 + eps)
    return torch.mean(out)

# Have a dictionary of keys to loss functions
LOSS_FNs = {
        'L1': L1,
        'L2': L2,
        'CE': CE,
}

def get_grid(image):
    ''' Return an affine grid in range [-1, 1] '''
    B = image.shape[0]
    grid = torch.FloatTensor([[1, 0, 0], [0, 1, 0]])[None]
    grid = grid.to(image.device)
    grid = grid.repeat(B, 1, 1)
    grid = F.affine_grid(grid, image.size(), align_corners=True)
    xx, yy = grid[..., 0], grid[..., 1]
    return xx, yy

def xytogrid(x, y):
    ''' Given x and y, concat to a grid '''
    grid = torch.cat([x[..., None], y[..., None]], -1)
    return grid

def small2largegrid(xx, yy, size):
    ''' Convert from [-1, 1] to [0, H] '''
    H, W = size[2:]
    xx = (xx + 1)/2 * W
    yy = (yy + 1)/2 * H
    return xx, yy

def large2smallgrid(xx, yy, size):
    ''' Convert from [0, H] to [-1, 1] '''
    H, W = size[2:]
    xx = xx * 2 / W - 1
    yy = yy * 2 / H - 1
    return xx, yy


def resample_from_flow_2d(image, flow):
    '''
    Given an image and flow vector, get the new image
    '''
    if type(flow) in [int, float] and flow == 0:
        return image

    B, C, H, W = image.shape
    # Resize this grid to [H, W] boundaries
    xx, yy = get_grid(image)
    xx, yy = small2largegrid(xx, yy, image.size())
    # Get new points
    xnew = xx + flow[:, 0]
    ynew = yy + flow[:, 1]
    # Convert these points back from [0, H] to [-1, 1] range
    xnew, ynew = large2smallgrid(xnew, ynew, image.size())
    # concat them back
    gridnew = xytogrid(xnew, ynew)
    imagenew = F.grid_sample(image+0, gridnew, align_corners=True)
    return imagenew


def flow_consistency_2d(parent, child):
    '''
    Given a parent flow, and a child flow, we will output the difference
    Basically, we will output p(x) + c(p(x))
    '''
    B, C, H, W = parent.shape
    size = parent.size()

    xx, yy = get_grid(parent)
    xx, yy = small2largegrid(xx, yy, size)
    # Find parent values
    xp = xx + parent[:, 0]
    yp = yy + parent[:, 1]
    # Given new coordinates, convert them and sample the values of children
    xp, yp = large2smallgrid(xp, yp, size)
    newgrid = xytogrid(xp, yp)
    newchild = F.grid_sample(child, newgrid, align_corners=True)
    # Given new child values, move as per that direction
    return parent + newchild


# Losses for vessel losses
def vessel_loss_2d(output, data, config):
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

    cosineabs = args.get('absolute_cosine', False)

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
        v1 = vessel[:, :2]
        v2 = vessel[:, 2:]
        # Intensity consistency loss
        if l_intensity:
            for scale in [0.2, 0.4, 0.6, 0.8, 1]:
                i_parent = resample_from_flow_2d(image, scale*v1)
                i_child = resample_from_flow_2d(image, scale*v2)
                if cosineabs:
                    # Check for both directions for same intensity -> this will help in centerline prediction
                    i_child2 = resample_from_flow_2d(image, -scale*v2)
                    L_childloss = torch.max(L_loss(image, i_child, mask=mask), L_loss(image, i_child2, mask=mask))
                    # add parent and child loss
                else:
                    L_childloss = L_loss(image, i_child, mask=mask)
                # Add that loss
                loss = loss + l_intensity * (L_loss(image, i_parent, mask=mask) + L_childloss)/5.0
        # Flow consistency loss
        if l_consistency:
            # If v1, v2 are supposed to be opposite directions
            if not cosineabs:
                loss = loss + l_consistency * L2(flow_consistency_2d(v1, v2), mask=mask)
            else:
                loss = loss + l_consistency * L2(flow_consistency_2d(v1, -v1), mask=mask)
        # Check for cosine similarity
        if l_cosine:
            if not cosineabs:
                loss = loss + l_cosine * (1 + (mask * F.cosine_similarity(v1, v2)).mean())  # adding 1 so that minimum value of loss is 0
            else:
                loss = loss + l_cosine * torch.abs(mask * F.cosine_similarity(v1, v2)).mean()
        # Check for decoder
        if l_decoder:
            loss = loss + l_decoder * L2(image, recon, mask=1)
        # Check for length of vector
        if l_length:
            v1norm = torch.sqrt((v1**2).sum(1) + eps)
            v2norm = torch.sqrt((v2**2).sum(1) + eps)
            loss = loss + l_length * (L1(1./v1norm, mask=mask) + L1(1./v2norm, mask=mask))

    else:
        raise NotImplementedError

    # Return loss
    return loss


def vessel_loss_2d_dampen(output, data, config):
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

    # parameter for followup vesselness
    l_followupv = args.get('lambda_followupv')

    # check if we want to minimize cross correlation
    #is_crosscorr = args.get('is_crosscorr', False)
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
        v1 = vessel[:, :2]
        v2 = vessel[:, 2:]
        # Intensity consistency loss
        if l_intensity:
            for scale in [0.2, 0.4, 0.6, 0.8, 1]:
                # Check for both directions for same intensity -> this will help in centerline prediction
                i_parent = resample_from_flow_2d(image, scale*v1)
                i_child = resample_from_flow_2d(image, scale*v2)
                i_child2 = resample_from_flow_2d(image, -scale*v2)
                L_childloss = torch.max(L_loss(image, i_child, mask=mask), L_loss(image, i_child2, mask=mask))
                # add parent and child loss
                # Add that loss
                loss = loss + l_intensity * (L_loss(image, i_parent, mask=mask) + L_childloss)/5.0

        # Flow consistency loss
        if l_consistency:
            # If v1, v2 are supposed to be opposite directions
            loss = loss + l_consistency * L2(flow_consistency_2d(v1, -v1), mask=mask)

        # Check for cosine similarity
        if l_cosine:
            loss = loss + l_cosine * torch.abs(mask * F.cosine_similarity(v1, v2)[:, None]).mean()

        # Check for decoder
        if l_decoder:
            loss = loss + l_decoder * L2(image, recon, mask=1)

        # Check for length of vector
        if l_length:
            v1norm = torch.sqrt((v1**2).sum(1) + eps)[:, None]
            loss = loss + l_length * L1(1./v1norm, mask=mask)

        # Check for length of vector for perpendicular line
        if l_perlength:
            v2norm = torch.sqrt((v2**2).sum(1) + eps)[:, None]
            loss = loss + l_perlength * L1(v2norm, mask=mask)

        # Check profile by taking convolution with the template [-1 -1 1 1 1 1 -1 -1]
        vessel_conv = 0.0
        if l_template:
            vessel_conv = v2vesselness(image, v2, num_samples_template, vessel_type, is_crosscorr=is_crosscorr)
            loss = loss + l_template * (1 - (mask*vessel_conv).mean())
            # Check image values
#             i_range = []
#             for s in np.linspace(-2, 2, num_samples_template):
#                 filt = 2*int(abs(s) < 1) - 1
#                 i_val = resample_from_flow_2d(image, s*v2)   # image is [B, 1, H, W], so is i_val
#                 # save it
#                 if detach:
#                     i_range.append(i_val.detach()[:, None])
#                 else:
#                     i_range.append(i_val[:, None])
#                 # Compute the convolution I * f
#                 vessel_conv = vessel_conv + i_val * filt

#             # Calculate std if cross correlation
#             if is_crosscorr:
#                 i_range = torch.cat(i_range, 1)   # [B, 20, 1, H, W]
#                 i_std = i_range.std(1) + 1e-10    # [B, 1, H, W]
#                 vessel_conv = vessel_conv / i_std

#             # Modify the vesselness according to parameters
#             if vessel_type == 'light':
#                 pass
#             elif vessel_type == 'dark':
#                 vessel_conv = -vessel_conv
#             elif vessel_type == 'both':
#                 vessel_conv = 2*torch.abs(vessel_conv)
#             else:
#                 raise NotImplementedError('{} keyword for vessel type is not supported'.format(vessel_type))

#             loss = loss + l_template * (1 - (mask*vessel_conv).mean()/num_samples_template)

        # Check for vesselness in followup
        if l_followupv and l_template:
            # we have already calculated the vesselness
            followup_vessel = resample_from_flow_2d(vessel_conv, v1)
            loss = loss + l_followupv * (1 - (mask*followup_vessel).mean()/num_samples_template)

    else:
        raise NotImplementedError

    # Return loss
    return loss




def vessel_loss_2d_sqmax(output, data, config):
    return vessel_loss_2d_sq(output, data, config, maxfilter=True)


def vessel_loss_2d_sq(output, data, config, maxfilter=False):
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

    # parameter for followup vesselness
    l_followupv = args.get('lambda_followupv')
    l_mi = args.get('lambda_mi')

    parallel_scale = args.get('parallel_scale', 2)

    # Check for max filter
    if maxfilter == False:
        vesselnessfun = v2_sq_vesselness
    else:
        vesselnessfun = v2_sqmax_vesselness

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
        v1 = vessel[:, :2]
        v2 = vessel[:, 2:]
        # Intensity consistency loss
        if l_intensity:
            for scale in [0.2, 0.4, 0.6, 0.8, 1]:
                # Check for both directions for same intensity -> this will help in centerline prediction
                i_parent = resample_from_flow_2d(image, scale*v1)
                i_child = resample_from_flow_2d(image, scale*v2)
                i_child2 = resample_from_flow_2d(image, -scale*v2)
                L_childloss = torch.max(L_loss(image, i_child, mask=mask), L_loss(image, i_child2, mask=mask))
                # add parent and child loss
                # Add that loss
                loss = loss + l_intensity * (L_loss(image, i_parent, mask=mask) + L_childloss)/5.0

        # Flow consistency loss
        if l_consistency:
            # If v1, v2 are supposed to be opposite directions
            loss = loss + l_consistency * L2(flow_consistency_2d(v1, -v1), mask=mask)

        # Check for cosine similarity
        if l_cosine:
            loss = loss + l_cosine * torch.abs(mask * F.cosine_similarity(v1, v2)[:, None]).mean()

        # Check for decoder
        if l_decoder:
            loss = loss + l_decoder * L2(image, recon, mask=1)

        # Check for length of vector
        if l_length:
            v1norm = torch.sqrt((v1**2).sum(1) + eps)[:, None]
            loss = loss + l_length * L1(1./v1norm, mask=mask)

        # Check for length of vector for perpendicular line
        if l_perlength:
            v2norm = torch.sqrt((v2**2).sum(1) + eps)[:, None]
            loss = loss + l_perlength * L1(v2norm, mask=mask)

        # Check profile by taking convolution with the template [-1 -1 1 1 1 1 -1 -1]
        vessel_conv = 0.0
        if l_template:
            vessel_conv = vesselnessfun(image, v2, num_samples_template, vessel_type, is_crosscorr=is_crosscorr, parallel_scale=parallel_scale)
            loss = loss + l_template * (1 - (mask*vessel_conv).mean())

        # Check for vesselness in followup
        if l_followupv and l_template:
            # we have already calculated the vesselness
            followup_vessel = resample_from_flow_2d(vessel_conv, v1)
            loss = loss + l_followupv * (1 - (mask*followup_vessel).mean())

        # Also check for mutual information
        if l_mi:
            mi = mutualinformation(image, vessel_conv, mask)
            loss = loss + (-mi)*l_mi

    else:
        raise NotImplementedError

    # Return loss
    return loss


def vessel_loss_2d_path(output, data, config):
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
    l_profile = args.get('lambda_profile')

    # Add extra parameters -> length and matching of template profile
    l_template = args.get('lambda_template')
    num_samples_template = args.get('num_samples_template', 10)
    l_perlength = args.get('lambda_perlength')
    detach = args.get('detach', True)
    # This parameter is for type of vessel
    vessel_type = config.get('vessel_type', 'dark') # should be light, dark or both

    # parameter for followup vesselness
    l_followupv = args.get('lambda_followupv')

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
        loss = 0.0 * torch.rand(1).squeeze().to(image.device)
        v1 = vessel[:, :2]
        v2 = vessel[:, 2:]
        # Intensity consistency loss
        if l_intensity:
            for scale in [0.2, 0.4, 0.6, 0.8, 1]:
                # Check for both directions for same intensity -> this will help in centerline prediction
                i_parent = resample_from_flow_2d(image, scale*v1)
                i_child = resample_from_flow_2d(image, scale*v2)
                i_child2 = resample_from_flow_2d(image, -scale*v2)
                L_childloss = torch.max(L_loss(image, i_child, mask=mask), L_loss(image, i_child2, mask=mask))
                # add parent and child loss
                # Add that loss
                loss = loss + l_intensity * (L_loss(image, i_parent, mask=mask) + L_childloss)/5.0

        # Flow consistency loss
        if l_consistency:
            # If v1, v2 are supposed to be opposite directions
            loss = loss + l_consistency * L2(flow_consistency_2d(v1, -v1), mask=mask)

        # Check for cosine similarity
        if l_cosine:
            #print_fn("before cosine", loss.data)
            loss = loss + l_cosine * torch.abs(mask * F.cosine_similarity(v1, v2)[:, None]).mean()

        # Check for decoder
        if l_decoder:
            #print_fn('before decoder', loss.data)
            loss = loss + l_decoder * L2(image, recon, mask=1)

        # Check for length of vector
        if l_length:
            v1norm = torch.sqrt((v1**2).sum(1) + eps)[:, None]
            loss = loss + l_length * L1(1./v1norm, mask=mask)

        # Check for length of vector for perpendicular line
        if l_perlength:
            v2norm = torch.sqrt((v2**2).sum(1) + eps)[:, None]
            loss = loss + l_perlength * L1(v2norm, mask=mask)

        # Check profile by taking convolution with the template [-1 -1 1 1 1 1 -1 -1]
        vessel_conv = 0.0
        if l_template:
            # here, we try to match two things independently, the first is the vessel profile
            #print_fn('before vesselness', loss.data)
            vessel_conv = v2vesselness(image, v2, num_samples_template, vessel_type, is_crosscorr=is_crosscorr)
            loss = loss + l_template * (1 - (mask*vessel_conv).mean())

            #print_fn('before path', loss.data)

            # The second is the similarity of profiles from nearby points (we will iteratively go on a path using v1)
            cur_profile = get_profile(image, v2, num_samples_template)

            # Given a v1, sample v2 at those locations, and take profiles at (v1 + __|---|__)
            # Then, add a step to v1
            cur_v1 = v1 + 0
            for i in range(10):
                v2_sample = resample_from_flow_2d(v2, cur_v1)
                new_profile = get_profile(image, v2_sample, num_samples_template, offset=cur_v1)
                crosscorr = get_cross_corr(cur_profile, new_profile)
                # Update loss function
                loss = loss + l_profile * (1 - crosscorr).mean() / 10.
                # Move to a new location of v1 now
                cur_v1 = cur_v1 + resample_from_flow_2d(v1, cur_v1)

            # Add a loss to maximize norm by a small amount
            loss = loss + l_length * 1./(1 + cur_v1**2).mean()
    else:
        raise NotImplementedError

    return loss


def v2_path_vesselness(image, v2, nsample=10, vtype='light', mask=None, percentile=100, is_crosscorr=False, v1=None):
    ''' Calculate a path based vesselness '''
    vessel_conv = v2vesselness(image, v2, nsample, vtype, is_crosscorr=is_crosscorr)
    vessel_conv = torch.max(vessel_conv, vessel_conv*0)
    H, W = image.shape[-2:]

    # Calculate the path
    cur_v1 = v1 + 0
    for i in range(10):
        cur_v1 = cur_v1 + resample_from_flow_2d(v1, cur_v1)

    path_displacement = (cur_v1[:, 0:1]**2 + cur_v1[:, 1:2]**2)/H/W*4.0   # Ranges from [0, 8]
    v = vessel_conv * path_displacement
    return v


def get_profile(image, v, n=20, scale=2, offset=0):
    ''' Get the image profile along a given direction '''
    i_range = []
    scale = abs(scale)
    for s in np.linspace(-scale, scale, n):
        ival = resample_from_flow_2d(image, s*v + offset)[:, None]
        i_range.append(ival)
    return torch.cat(i_range, 1)  # [B, n, C, H, W]


def get_cross_corr(p1, p2):
    ''' Given two profiles, return cross correlation '''
    m1 = p1.mean(1, keepdim=True)
    m2 = p2.mean(1, keepdim=True)
    s1 = p1.std(1, keepdim=True).detach() + 1e-2
    s2 = p2.std(1, keepdim=True).detach() + 1e-2
    # Calculate crosscorr
    corr = (p1 - m1)*(p2 - m2)/s1/s2
    corr = corr.mean(1)
    return corr


"""
####################################################################################
HERE IS THE CURVED LOSS
####################################################################################
"""
# Output vesselness over variational filters
def v2_curved_vesselness(image, ves, nsample=12, vtype='light', mask=None, percentile=100, is_crosscorr=False, v1=None, parallel_scale=(4, 10)):
    response = 0.0
    i_range = []

    # Parallel direction
    v1 = ves*0
    sign = 2*(v1[:, 0] >= 0) - 1
    sign = sign.detach()
    v1[:, 1] = ves[:, 0] * sign
    v1[:, 0] = -ves[:, 1] * sign

    # Perpendicular direction
    v2 = ves

    # The parallel scale is a two-pair of scale and nsample
    p_totalscale, p_sample = parallel_scale
    p_scale = p_totalscale*1.0/p_sample

    cur_v1 = 0
    for i in range(p_sample):
        # sample v1 at appropriate location
        cur_v2 = resample_from_flow_2d(v2, cur_v1)
        # Calculate vesselness at this profile
        for s in np.linspace(-2, 2, nsample):
            filt = 2*int(abs(s) <= 1) - 1
            i_val = resample_from_flow_2d(image, cur_v1 + s*cur_v2)
            if is_crosscorr:
                i_range.append(i_val.detach()[:, None])
            # Compute response
            response = response + (i_val*filt)

        # Compute next v1
        next_v1 = resample_from_flow_2d(v1, cur_v1)
        # Update the v1 direction
        cur_v1 = cur_v1 + p_scale*next_v1

    # Take mean of response
    response = response/(nsample*p_sample)
    # Normalize
    if is_crosscorr:
        i_range = torch.cat(i_range, 1)   # [B, 20, 1, H, W]
        i_std = i_range.std(1, unbiased=False) + 1e-5    # [B, 1, H, W]
        response = response / i_std

        # Check assertion
        idx = (-1 <= response)*(response <= 1)
        idx = (~idx).sum().item()
        assert idx == 0, '{} {}'.format(idx, np.prod(list(response.shape)))

    # Correct the response accordingly
    if vtype == 'light':
        pass
    elif vtype == 'dark':
        response = -response
    elif vtype == 'both':
        response = torch.abs(response)
    else:
        raise NotImplementedError('{} type not supported in vesseltype'.format(vtype))

    # We got the response, now subtract from mean and multiply with optional mask
    response = response - response.min().detach()
    if mask is not None:
        response = response * mask
    return response



# Auxiliary class for actually calculating the ODE
class ODEVesselness(nn.Module):
    def __init__(self, v1, v2, image, is_crosscorr, nsample=12):
        super().__init__()
        self.v1 = v1
        self.v2 = v2
        self.image = image
        self.is_crosscorr = is_crosscorr
        self.nsample = nsample


    def forward(self, t, data):
        # Data is of size [B, 3, H, W] where first channel is vesselness, other 2 channels are x, y
        v = data[:, :1]
        x = data[:, 1:]
        # Calculate the derivatives based on displacement
        v2_x = resample_from_flow_2d(self.v2, x)
        i_range1 = []
        i_range2 = []
        res1, res2 = 0, 0
        D = 0
        # Calculate response
        for s in np.linspace(-2, 2, self.nsample):
            filt = 2*int(abs(s) < 1) - 1
            D += filt
            i_val = resample_from_flow_2d(self.image, s*v2_x + x)
            if s < 0:
                i_range1.append(i_val.detach()[:, None])
            else:
                i_range2.append(i_val.detach()[:, None])
            # Compute the convolution I * f
            if s < 0:
                res1 = res1 + (i_val * filt)
            else:
                res2 = res2 + (i_val * filt)

        assert D == 0
        # Currently only using dark vessels
        if self.is_crosscorr:
            i_range1 = torch.cat(i_range1, 1)
            i_range2 = torch.cat(i_range2, 1)
            i_std1 = i_range1.std(1) + 1e-5
            i_std2 = i_range2.std(1) + 1e-5
            res1 = res1 / i_std1
            res2 = res2 / i_std2

        res = torch.min(-res1, -res2)
        # Calculate rate of change of x, given by v1 at that location
        v1_x = resample_from_flow_2d(self.v1, x)
        dx = torch.cat([res, v1_x], 1)
        return dx


def v2_ode_vesselness(image, ves, nsample=12, vtype='light', mask=None, percentile=100, is_crosscorr=False, v1=None, parallel_scale=(4, 50)):
    # Use ode library
    v2 = ves
    v1 = ves*0
    v1[:, 0] = -ves[:, 1] + 0
    v1[:, 1] =  ves[:, 0] + 0
    # Modify sign
    sign = 2*(v1[:, :1].detach() > 0) - 1
    v1 = v1 * sign

    # Given v1, we can use it to move along
    func = ODEVesselness(v1, v2, image, is_crosscorr, nsample)
    Nscale, timestep = parallel_scale
    B, _, H, W = image.shape
    vxy_0 = torch.zeros((B, 3, H, W)).to(image.device)
    vxy_t1 = odeint(func, vxy_0, torch.linspace(0, Nscale, 2).to(image.device), rtol=1e-2)  # Going forwards
    vxy_t2 = odeint(func, vxy_0, torch.linspace(0, -Nscale, 2).to(image.device), rtol=1e-2)  # Going backwards, note that this vesselness will be negative
    v1 = vxy_t1[-1, :, :1]
    v2 = -vxy_t2[-1, :, :1]
    v = (v1 + v2)/2.0
    v = v - v.min().detach()
    if mask is not None:
        v = v * mask
    return v


####
## CURVED VESSEL LOSS FUNCTIONS
####


def vessel_loss_2d_ode(output, data, config):
    return vessel_loss_2d_curved(output, data, config, ode=True)


# Now the actual loss which uses the vesselness
def vessel_loss_2d_curved(output, data, config, ode=False):
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

    # parameter for followup vesselness
    l_followupv = args.get('lambda_followupv')

    parallel_scale = args.get('parallel_scale')

    # check if we want to minimize cross correlation
    is_crosscorr = args.get('is_crosscorr', False)
    if not ode:
        v2func = v2_curved_vesselness
    else:
        v2func = v2_ode_vesselness

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
        v1 = vessel[:, :2]
        v2 = vessel[:, 2:]
        # Intensity consistency loss
        if l_intensity:
            for scale in [0.2, 0.4, 0.6, 0.8, 1]:
                # Check for both directions for same intensity -> this will help in centerline prediction
                i_parent = resample_from_flow_2d(image, scale*v1)
                i_child = resample_from_flow_2d(image, scale*v2)
                i_child2 = resample_from_flow_2d(image, -scale*v2)
                L_childloss = torch.max(L_loss(image, i_child, mask=mask), L_loss(image, i_child2, mask=mask))
                # add parent and child loss
                # Add that loss
                loss = loss + l_intensity * (L_loss(image, i_parent, mask=mask) + L_childloss)/5.0

        # Flow consistency loss
        if l_consistency:
            # If v1, v2 are supposed to be opposite directions
            loss = loss + l_consistency * L2(flow_consistency_2d(v1, -v1), mask=mask)

        # Check for cosine similarity
        if l_cosine:
            loss = loss + l_cosine * torch.abs(mask * F.cosine_similarity(v1, v2)[:, None]).mean()

        # Check for decoder
        if l_decoder:
            loss = loss + l_decoder * L2(image, recon, mask=1)

        # Check for length of vector
        if l_length:
            v1norm = torch.sqrt((v1**2).sum(1) + eps)[:, None]
            loss = loss + l_length * L1(1./v1norm, mask=mask)

        # Check for length of vector for perpendicular line
        if l_perlength:
            v2norm = torch.sqrt((v2**2).sum(1) + eps)[:, None]
            loss = loss + l_perlength * L1(v2norm, mask=mask)

        # Check profile by taking convolution with the template [-1 -1 1 1 1 1 -1 -1]
        vessel_conv = 0.0
        if l_template:
            vessel_conv = v2func(image, v2, num_samples_template, vessel_type, is_crosscorr=is_crosscorr, parallel_scale=parallel_scale)
            loss = loss + l_template * (1 - (mask*vessel_conv).mean())

    else:
        raise NotImplementedError

    # Return loss
    return loss


def mutualinformation(image, ves, mask=None, bins=None, sigma_factor=0.5, epsilon=0):
    # Global mutual information
    if bins is None:
        bins = np.linspace(-1, 1, 20)
    # Get number of bins, sigma and normalizer
    numbins = len(bins)
    sigma = np.mean(np.abs(np.diff(bins))) * sigma_factor
    preterm = 1./2/sigma**2

    # Convert to torch tensor
    bins = torch.FloatTensor(bins).to(image.device)[None]  # [1, B]

    mi = 0.0
    B, C, H, W = image.shape
    if mask is not None:
        # take each image and vessel with mask
        for i in range(B):
            c, y, x = torch.where(mask[i] > 0.5)
            img = image[i, c, y, x][:, None]         # [HW, 1]
            v = ves[i, c, y, x][:, None]             # [HW, 1]

            # Given these, find p(a) and p(b)
            pa = 1e-10 + torch.exp(-preterm * (img - bins)**2)   # [HW, B]
            pa = pa / pa.sum(-1, keepdims=True)

            pb = 1e-10 + torch.exp(-preterm * (v - bins)**2)   # [HW, B]
            pb = pb / pb.sum(-1, keepdims=True)

            # Get p(a, b)
            pab = torch.mm(pa.T, pb)  # [B, B]
            pab = pab/pa.shape[0]

            pa = pa.mean(0, keepdims=True)  #[1, B]
            pb = pb.mean(0, keepdims=True)  #[1, B]
            papb = torch.mm(pa.T, pb) + epsilon #[B, B]
            _mival = torch.sum(pab * torch.log(epsilon + pab/papb))
            mi = mi + _mival

        mi = mi/B
        return mi
    else:
        raise NotImplementedError


def v1_sq_vesselness_test(image, output, nsample=12, vtype='light', mask=None, percentile=100, is_crosscorr=False, v1 = None, parallel_scale=2):
    v = v1_sq_vesselness(image, output, nsample, vtype, mask, percentile, is_crosscorr, v1, parallel_scale)
    #v = torch.exp(v)
    ves = output['vessel'][:, 2:4]

    #### Additional dissimilarity
    total_sim = 0.0
    for sv in np.linspace(-parallel_scale*4, parallel_scale*4, nsample):
        vt = resample_from_flow_2d(ves+0, sv*ves)
        sim = (torch.abs(F.cosine_similarity(ves+0, vt)))
        total_sim = total_sim + sim

    total_sim = total_sim/nsample
    total_sim = total_sim[:, None]

    #return v + 0.05*total_sim
    return v*total_sim


def v1_sq_vesselness(image, output, nsample=12, vtype='light', mask=None, percentile=100, is_crosscorr=False, v1 = None, parallel_scale=2):
    response1 = 0.0
    response2 = 0.0
    i_range1 = []
    i_range2 = []

    ves = output['vessel'][:, 2:4]

    v2 = ves*0
    v2[:, 1] = ves[:, 0] + 0
    v2[:, 0] = -ves[:, 1] + 0

    D = 0
    N = nsample*nsample

    for sv in np.linspace(-parallel_scale, parallel_scale, nsample):
        # Get perp profile
        for s in np.linspace(-2, 2, nsample):
            filt = 2*int(abs(s) < 1) - 1
            D += filt
            i_val = resample_from_flow_2d(image, sv*ves + s*v2)
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

    response = response1 + response2

    if is_crosscorr:
        # Take min of both sides
        i_range = torch.cat(i_range1 + i_range2, 1)
        i_std = i_range.std(1, unbiased=True) + 1e-5
        response = response / i_std / N
    else:
        response = response / N

    # Correct the response accordingly
    if vtype == 'light':
        pass
    elif vtype == 'dark':
        response = -response
    elif vtype == 'both':
        response = 2*torch.abs(response)
    else:
        raise NotImplementedError('{} type not supported in vesseltype'.format(vtype))

    # We got the response, now subtract from mean and multiply with optional mask
    if mask is not None:
        response = response * mask
    return response


def v1_sqmax_vesselness_test(image, output, nsample=12, vtype='light', mask=None, percentile=100, is_crosscorr=False, v1 = None, parallel_scale=2):
    v = v1_sqmax_vesselness(image, output, nsample, vtype, mask, percentile, is_crosscorr, v1, parallel_scale)
    #v = torch.exp(v)

    ves = output['vessel'][:, 2:4]

    #### Additional dissimilarity
    total_sim = 0.0
    for sv in np.linspace(-parallel_scale*4, parallel_scale*4, nsample):
        vt = resample_from_flow_2d(ves+0, sv*ves)
        sim = torch.abs(F.cosine_similarity(ves+0, vt))
        total_sim = total_sim + sim

    total_sim = total_sim/nsample
    total_sim = total_sim[:, None]

    #return v + 0.05*total_sim
    return v*total_sim


def v1_sqmax_vesselness(image, output, nsample=12, vtype='light', mask=None, percentile=100, is_crosscorr=False, v1 = None, parallel_scale=2, sv_range=None):
    '''
    Parameter list:
    image: Image on which vesselness needs to be calculated
    output: dictionary of all outputs, vessel is in the keyword
    nsample: number of datapoints to sample from each side
    vtype: vessel type (light, dark or both)
    mask: optional mask to mask ROI for vesselness
    percentile: deprecated
    is_crosscorr: Should we use cross-correlation or not? Might help for detecting low-contrast vessels as well
    v1: deprecated, calculated from output
    parallel_scale: Scale upto which it should be calculated
    sv_range: A range for calculating the extent of 'v1' (can be useful for bifurcations)
    '''
    response1 = 0.0
    response2 = 0.0
    i_range1 = []
    i_range2 = []

    ves = output['vessel'][:, 2:4]

    v2 = ves*0
    v2[:, 1] = ves[:, 0] + 0
    v2[:, 0] = -ves[:, 1] + 0

    D = 0
    N = nsample*nsample/2

    # Default range is [-parallel_scale, parallel_scale]
    if sv_range is None:
        sv_range = (-parallel_scale, parallel_scale)
    assert len(sv_range) == 2

    for sv in np.linspace(sv_range[0], sv_range[1], nsample):
        # Get perp profile
        for s in np.linspace(-2, 2, nsample):
            filt = 2*int(abs(s) < 1) - 1
            D += filt
            i_val = resample_from_flow_2d(image, sv*ves + s*v2)
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
        i_range1 = torch.cat(i_range1, 1)
        i_std1 = i_range1.std(1, unbiased=False) + 1e-5
        response1 = response1 / i_std1 / N

        i_range2 = torch.cat(i_range2, 1)
        i_std2 = i_range2.std(1, unbiased=False) + 1e-5
        response2 = response2 / i_std2 / N
    else:
        response1 = response1 / N
        response2 = response2 / N

    # Correct the response accordingly
    if vtype == 'light':
        response = torch.min(response1, response2)
    elif vtype == 'dark':
        response = torch.min(-response1, -response2)
    elif vtype == 'both':
        response = torch.min(torch.abs(response1), torch.abs(response2))
    else:
        raise NotImplementedError('{} type not supported in vesseltype'.format(vtype))

    # We got the response, now subtract from mean and multiply with optional mask
    if mask is not None:
        response = response * mask
    return response


def rotate_vector_2d(v1, theta):
    '''
    v1 is a vector of size [B, 2N, H, W] containing (x1, y1), ... (xn, yn)
    theta is a vector of size [B, 1, H, W] containing angle in radians
    '''
    N = v1.shape[1]
    tN = theta.shape[1]
    assert tN == 1
    assert N%2 == 0
    # Now edit the vectors
    outv = torch.zeros_like(v1, device=v1.device)
    cos = torch.cos(theta).squeeze()
    sin = torch.sin(theta).squeeze()
    # For each vector, put the appropriate value
    for i in range(N//2):
        x = v1[:, 2*i]
        y = v1[:, 2*i+1]
        # Get rotated coordinates
        X = x*cos - y*sin
        Y = x*sin + y*cos
        # save to new vector
        outv[:, 2*i] = X
        outv[:, 2*i+1] = Y
    outv = outv.contiguous()
    return outv


def v1_sqmax_bifurc(image, output, nsample=12, vtype='light', mask=None, percentile=100, is_crosscorr=False, v1 = None, parallel_scale=2, sv_range=None):
    '''
    Compute bifurcation vesselness separately
    '''
    output1 = {}
    output2 = {}
    theta1 = -output['bangle'][:, 0:1]
    theta2 =  output['bangle'][:, 1:2]
    # Get vessel direction
    v1 = output['vessel']
    # Get vb1 and vb2
    vbranch1 = rotate_vector_2d(-v1, theta1)
    vbranch2 = rotate_vector_2d(-v1, theta2)
    # Put them into new dicts
    output1['vessel'] = vbranch1
    output2['vessel'] = vbranch2
    # Put them into sq vesselness
    v_straight = v1_sqmax_vesselness(image, output, nsample, vtype, mask, percentile, is_crosscorr, None, parallel_scale, (0, parallel_scale))
    v_b1 = v1_sqmax_vesselness(image, output1, nsample, vtype, mask, percentile, is_crosscorr, None, parallel_scale, (0, parallel_scale))
    v_b2 = v1_sqmax_vesselness(image, output2, nsample, vtype, mask, percentile, is_crosscorr, None, parallel_scale, (0, parallel_scale))
    return (v_straight + v_b1 + v_b2)/3.0


def v1_sqmax_jointvesselness(image, output, nsample=12, vtype='light', mask=None, percentile=100, is_crosscorr=False, v1 = None, parallel_scale=2, sv_range=None):
    '''
    Joint vesselness using both normal vessels and bifurcations
    '''
    normalv = v1_sqmax_vesselness(image, output, nsample, vtype, mask, percentile, is_crosscorr, None, parallel_scale, sv_range)
    bifurcv = v1_sqmax_bifurc(image, output, nsample, vtype, mask, percentile, is_crosscorr, None, parallel_scale, sv_range)
    bwt = output['bwt']
    return bwt*bifurcv + (1 - bwt)*normalv


def vessel_loss_2dv1_sq(output, data, config):
    return vessel_loss_2dv1_sqmax(output, data, config, maxfilter=False)


def vessel_loss_2dv1_bifurcmax(output, data, config):
    '''
    Loss with bifurcation consideration
    '''
    args = config['loss_args']
    bifurc_mode = args.get('bifurc_mode')
    assert bifurc_mode in ['joint', 'detach']
    return vessel_loss_2dv1_sqmax(output, data, config)


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

    # Bifurcation parameters
    bifurc_mode = args.get('bifurc_mode')
    l_bifurcwt = args.get('lambda_bifurcwt')

    # Check for max filter
    if maxfilter==False:
        vesselnessfun = v1_sq_vesselness
        raise NotImplementedError
    else:
        vesselnessfun = v1_sqmax_vesselness             # This function is only for vesselness
        bifurcfun = v1_sqmax_bifurc                     # This function is for bifurcation only
        jointvesselnessfun = v1_sqmax_jointvesselness   # This function combines both methods with weighted average

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
    loss = 0.0
    if num_dir == 2 and not unc:
        assert config['arch']['args']['out_channels'] == 4, 'Model with 2 directions and no uncertainty'
        # parameters are v1, v2
        loss = 0.0
        v1 = vessel[:, 2:4]
        # Consistency loss
        if l_consistency:
            # Align v1 = v(x1) and v2 = v(x1 + v1) to be along same directions
            v1x = resample_from_flow_2d(v1+0, v1+0)
            loss = loss - l_consistency * F.cosine_similarity(v1, v1x).mean()

        # Check for cosine similarity
        if l_cosine:
            v1x = resample_from_flow_2d(v1+0, v1+0)
            sim = F.cosine_similarity(v1, v1x)
            sim0 = sim*0
            loss = loss - l_consistency * torch.min(sim, sim0).mean()

        # Check for decoder
        if l_decoder:
            loss = loss + l_decoder * L2(image, recon, mask=1)

        # Check profile by taking convolution with the template [-1 -1 1 1 1 1 -1 -1]
        vessel_conv = 0.0
        if l_template:
            vessel_conv = vesselnessfun(image, output, num_samples_template, vessel_type, is_crosscorr=is_crosscorr, parallel_scale=parallel_scale)
            # Calculate loss depending on what bifurcation mode we choose
            if bifurc_mode is None:
                loss = loss + l_template * (1 - (mask*vessel_conv).mean())
            else:
                # There is a bifurcation mode (either joint or detach)
                if bifurc_mode == 'joint':
                    # Joint mode, take weighted average of vesselness and optimize this directly
                    jointv = jointvesselnessfun(image, output, num_samples_template, vessel_type, is_crosscorr=is_crosscorr, parallel_scale=parallel_scale)
                    loss = loss + l_template * (1 - (mask*jointv).mean())
                else:
                    bifurcves = bifurcfun(image, output, num_samples_template, vessel_type, is_crosscorr=is_crosscorr, parallel_scale=parallel_scale)
                    # Add bifurc loss and normal vessels separately
                    loss = loss + l_template * (1 - (mask*vessel_conv).mean())
                    loss = loss + l_template * (1 - (mask*bifurcves).mean())
                    # Compute weights of balancing parameter 'b'
                    bwt = output['bwt']
                    bwt_loss =  bwt * bifurcves.detach() + (1 - bwt) * vessel_conv.detach()
                    # Calculate loss
                    loss = loss + l_bifurcwt * (1 - (mask*bwt_loss).mean())


        # Vessel intensity consistency loss
        if l_intensity:
            if bifurc_mode is None:
                lower = -parallel_scale/2.0
            else:
                lower = 0
            # If there are bifurcations in consideration, then only move along the lower half
            for scale in np.linspace(lower, parallel_scale/2.0, 5):
                # Check for both directions for same intensity -> this will help in centerline prediction
                i_parent = resample_from_flow_2d(vessel_conv, scale*v1)
                loss = loss + l_intensity * (L_loss(vessel_conv, i_parent, mask=mask))/5.0

    else:
        raise NotImplementedError

    # Return loss
    return loss
