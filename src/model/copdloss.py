import numpy as np
import itertools
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


def L1(i1, i2=0, mask=1):
    return (torch.abs(i1 - i2)*mask).mean()

def L2(i1, i2=0, mask=1):
    return torch.mean(mask * (i1 - i2)**2)

def CE(i1, i2, eps=1e-10):
    out = -i1*torch.log(i2 + eps) - (1 - i1)*torch.log(1 - i2 + eps)
    return torch.mean(out)

# Have a dictionary of keys to loss functions
LOSS = {
        'L1': L1,
        'L2': L2,
        'CE': CE,
}

def null_fn(*args):
    return None

#print_fn = print
print_fn = null_fn

def get3dgrid(image):
    ''' Return an affine grid in range [-1, 1] '''
    B = image.shape[0]
    grid = torch.FloatTensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])[None]
    grid = grid.to(image.device)
    grid = grid.repeat(B, 1, 1)
    grid = F.affine_grid(grid, image.size(), align_corners=True)
    return grid  # [B, H, W, D, 3]

def largetosmall3d(Grid, H, W, D):
    # grid is of size [B, H, W, D, 3]
    # Convert [0, H] scale to [-1, 1] scale
    grid = Grid + 0
    grid[..., 0] = 2.0*grid[..., 0]/H - 1
    grid[..., 1] = 2.0*grid[..., 1]/W - 1
    grid[..., 2] = 2.0*grid[..., 2]/D - 1
    return grid

def smalltolarge3d(Grid, H, W, D):
    # grid is of size [B, H, W, D, 3]
    # Convert [-1, 1] scale to [0, H] scale
    grid = Grid + 0
    grid = grid*0.5 + 0.5
    grid[..., 0] = grid[..., 0]*H
    grid[..., 1] = grid[..., 1]*W
    grid[..., 2] = grid[..., 2]*D
    return grid

def resample_from_flow_3d(image, flow):
    # Image is [B, C, H, W, D]
    # Flow is [B, 3, H, W, D]
    B, _, H, W, D = image.shape
    grid = get3dgrid(image)  # BHWD 3  in [-1, 1]
    grid = smalltolarge3d(grid, H, W, D)  # Convert to [0, H]
    r_flow = flow.permute(0, 2, 3, 4, 1) + 0  # BHWD 3
    # Add it to flow
    grid = grid + r_flow  # now use this grid to resample
    grid = largetosmall3d(grid, H, W, D)  # Convert to [-1, 1]
    image_resample = F.grid_sample(image+0, grid, align_corners=True)
    return image_resample


def normalize_vector(v):
    # Normalize the vector to have unit norm (check dimension 1)
    assert v.shape[1] == 3
    norm = v.norm(dim=1, keepdim=True) + 1e-10
    return v / norm


def get_orthogonal_basis(v1, scale_vector=True):
    # Get an orthogonal basis with optional scaling added to it
    if scale_vector:
        scale = v1.norm(dim=1, keepdim=True)
    else:
        scale = 1

    '''
    c = torch.abs(v1)
    c = c - torch.min(v1, 1, keepdim=True).values
    c = (c <= 0).float()
    c = c.detach()
    '''
    c = torch.randn_like(v1, device=v1.device)
    v2 = scale*normalize_vector(torch.cross(c, v1, dim=1))
    v3 = scale*normalize_vector(torch.cross(v1, v2, dim=1))
    return v2, v3


def get_acute_projection(v, v1):
    '''
    Get the projection of v where v1 is the reference, also normalize it
    '''
    sign = F.cosine_similarity(v, v1)[:, None].detach()
    sign = (sign >= 0).float()*2 - 1
    v = sign*v
    v = normalize_vector(v)
    # Get scale
    scale = v1.norm(dim=1, keepdim=True)
    v = scale*v
    return v


def v13d_sq_jointvesselness(image, output, nsample=12, vtype='light', mask=None, percentile=100, is_crosscorr=False, parallel_scale=2, sv_range=None):
    '''
    Consider 3 directions, the main vessel direction, and 2 extra directions for birfurcations
    '''
    output1 = {}
    output2 = {}
    v1 = output['vessel'][:, :3]
    branch1 = get_acute_projection(output['branch1'], -v1)
    branch2 = get_acute_projection(output['branch2'], -v1)
    # Get dicts from v1
    output1['vessel'] = branch1
    output2['vessel'] = branch2
    # Get vesselness at different sv_range
    v1 = v13d_sq_vesselness(image, output, nsample, vtype, mask, percentile, is_crosscorr, parallel_scale, (0, parallel_scale))
    v2 = v13d_sq_vesselness(image, output1, nsample, vtype, mask, percentile, is_crosscorr, parallel_scale, (0, parallel_scale))
    v3 = v13d_sq_vesselness(image, output2, nsample, vtype, mask, percentile, is_crosscorr, parallel_scale, (0, parallel_scale))
    return (v1 + v2 + v3)/3.0


def v13d_sq_vesselness(image, output, nsample=12, vtype='light', mask=None, percentile=100, is_crosscorr=False, parallel_scale=2, sv_range=None):
    # Take actual Hessian from the direction specified by v1
    # Find v2 and v3 first
    # Create a v2 using the minimum abs value from v1 index
    v1 = output['vessel'][:, :3]
    v2, v3 = get_orthogonal_basis(v1)

    # Get default sv_range
    if sv_range is None:
        sv_range = (-parallel_scale, parallel_scale)

    # Keep track of all values, and add to response
    i_range = []
    response = 0.0
    for sv in np.linspace(sv_range[0], sv_range[1], nsample):
        # Get perpendicular profile
        for ang, s in itertools.product(np.arange(4), np.linspace(-2, 2, nsample)):
            # Calculate filter, theta, and actual direction
            filt = 2*int(abs(s) < 1) - 1
            theta = np.pi/4*ang
            vperp = np.cos(theta)*v2 + np.sin(theta)*v3
            # Get image
            i_val = resample_from_flow_3d(image, s*vperp + sv*v1)
            if is_crosscorr:
                i_range.append(i_val.detach()[:, None])
            # Add to response
            response = response + (i_val * filt)

    if is_crosscorr:
        i_range = torch.cat(i_range, 1)
        i_std = i_range.std(1, unbiased=False) + 1e-7
        response = response / i_std / (nsample**2) / 4
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

    return response


def v13d_sq_vesselness_test(image, output, nsample=12, vtype='light', mask=None, percentile=100, is_crosscorr=False, parallel_scale=2, sv_range=None):
    ''' Put additional dissimilarity term '''
    v = v13d_sq_vesselness(image, output, nsample, vtype, mask, percentile, is_crosscorr, parallel_scale, sv_range)
    ves = output['vessel'][:, :3]
    #### Additional dissimilarity
    total_sim = 0.0
    for sv in np.linspace(-parallel_scale*4, parallel_scale*4, nsample):
        vt = resample_from_flow_3d(ves+0, sv*ves)
        sim = (torch.abs(F.cosine_similarity(ves+0, vt)))
        total_sim = total_sim + sim

    total_sim = total_sim/nsample
    total_sim = total_sim[:, None]

    return v*total_sim


def v13d_sqmax_vesselness_test(image, output, nsample=12, vtype='light', mask=None, percentile=100, is_crosscorr=False, parallel_scale=2, sv_range=None):
    ''' Put additional dissimilarity term '''
    v = v13d_sqmax_vesselness(image, output, nsample, vtype, mask, percentile, is_crosscorr, parallel_scale, sv_range)
    ves = output['vessel'][:, :3]
    #### Additional dissimilarity
    total_sim = 0.0
    for sv in np.linspace(-parallel_scale*4, parallel_scale*4, nsample):
        vt = resample_from_flow_3d(ves+0, sv*ves)
        sim = (torch.abs(F.cosine_similarity(ves+0, vt)))
        total_sim = total_sim + sim

    total_sim = total_sim/nsample
    total_sim = total_sim[:, None]

    return v*total_sim


def v13d_sqmax_vesselness(image, output, nsample=12, vtype='light', mask=None, percentile=100, is_crosscorr=False, parallel_scale=2, sv_range=None):
    # Take actual Hessian from the direction specified by v1
    # Find v2 and v3 first
    # Create a v2 using the minimum abs value from v1 index
    v1 = output['vessel'][:, :3]
    v2, v3 = get_orthogonal_basis(v1)

    # Keep track of all values, and add to appropriate cross section
    response = [0.0]*8
    i_range = []
    for i in range(8):
        i_range.append([])

    # For each position, add to profile
    for sv in np.linspace(-parallel_scale, parallel_scale, nsample):
        # Get perpendicular profile
        for ang, s in itertools.product(np.arange(4), np.linspace(-2, 2, nsample)):
            # Calculate filter, theta, and actual direction
            filt = 2*int(abs(s) < 1) - 1
            theta = np.pi/4*ang
            vperp = np.cos(theta)*v2 + np.sin(theta)*v3
            # Get image
            i_val = resample_from_flow_3d(image, s*vperp + sv*v1)

            ## Determine index
            idx = ang + 4*int(s < 0)
            response[idx] = response[idx] + i_val*filt
            if is_crosscorr:
                i_range[idx].append(i_val.detach()[:, None])

    if is_crosscorr:
        for idx in range(8):
            # Compute std
            i_range[idx] = torch.cat(i_range[idx], 1)
            i_std = i_range[idx].std(1, unbiased=False) + 1e-7
            # Divide correlation by std
            response[idx] = response[idx] / i_std / (nsample**2) * 2
            # Check assertion
            val = (-1 <= response[idx])*(response[idx] <= 1)
            val = (~val).sum().item()
            assert val == 0, '{} {}'.format(val, np.prod(list(response.shape)))

    # Take weakest response over a ridge
    weakestresponse = [0.0]*8
    for i in range(8):
        for j in range(i, i+4):
            weakestresponse[i] = weakestresponse[i] + response[j%8][None]/4.0

    # Size [8, Batch, C, H, W, D]
    weakestresponse = torch.cat(weakestresponse, 0)

    # Correct the response accordingly
    if vtype == 'light':
        weakestresponse = torch.min(weakestresponse, 0).values
    elif vtype == 'dark':
        weakestresponse = torch.min(-weakestresponse, 0).values
    elif vtype == 'both':
        weakestresponse = torch.min(2*torch.abs(weakestresponse), 0).values
    else:
        raise NotImplementedError('{} type not supported in vesseltype'.format(vtype))

    return weakestresponse


def vessel_loss_3d(output, data, config,):
    return vessel_loss_3dmax(output, data, config, False)


def vessel_loss_3d_bifurc(output, data, config):
    '''
    Vessel birfurc with average from everywhere
    '''
    args = config['loss_args']
    bifurc_mode = args.get('bifurc_mode')
    assert bifurc_mode == 'only'
    return vessel_loss_3d(output, data, config,)


def vessel_loss_3dmax(output, data, config, maxfilter=True):
    '''
    Master loss function of vessel self supervised learning
    '''
    args = config['loss_args']
    # Get all parameters
    L_loss = LOSS[args['loss_intensity']]
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
    # l_bifurcwt = args.get('lambda_bifurcwt')
    sv_range = args.get('sv_range')

    # Check for max filter
    if maxfilter==False:
        vesselnessfun = v13d_sq_vesselness
        jointvesselfun = v13d_sq_jointvesselness
    else:
        vesselnessfun = v13d_sqmax_vesselness

    # check if we want to minimize cross correlation
    is_crosscorr = args.get('is_crosscorr', True)

    # Get outputs and inputs
    vessel = output['vessel']
    recon  = output['recon']
    image = data['image']

    # Take mask for all losses
    mask = data.get('mask', 1)
    if not args.get('use_mask', False):
        mask = 1

    # Now use the losses given in the config
    loss = 0.0
    v1 = vessel[:, :3]
    # Consistency loss
    if l_consistency:
        # Align v1 = v(x1) and v2 = v(x1 + v1) to be along same directions
        v1x = resample_from_flow_3d(v1+0, v1+0)
        loss = loss - l_consistency * F.cosine_similarity(v1, v1x).mean()

    # Decoder loss
    if l_decoder:
        loss = loss + l_decoder * L2(recon, image)

    # Check for cosine similarity
    if l_cosine:
        v1x = resample_from_flow_3d(v1+0, v1+0)
        sim = F.cosine_similarity(v1, v1x)
        sim0 = sim*0
        loss = loss - l_consistency * torch.min(sim, sim0).mean()

    # Check profile by taking convolution with the template
    vessel_conv = 0.0
    if l_template:
        if bifurc_mode is None:
            vessel_conv = vesselnessfun(image, output, num_samples_template, vessel_type, is_crosscorr=is_crosscorr, parallel_scale=parallel_scale)
            loss = loss + l_template * (1 - (mask*vessel_conv).mean())
        else:
            vessel_conv = jointvesselfun(image, output, num_samples_template, vessel_type, is_crosscorr=is_crosscorr, parallel_scale=parallel_scale)
            loss = loss + l_template * (1 - (mask*vessel_conv).mean())

    # Vessel intensity consistency loss
    if l_intensity:
        if bifurc_mode is None:
            lower = -parallel_scale/2.0
        else:
            lower = 0

        for scale in np.linspace(lower, parallel_scale/2., 5):
            # Check for both directions for same intensity -> this will help in centerline prediction
            i_parent = resample_from_flow_3d(vessel_conv, scale*v1)
            loss = loss + l_intensity * (L_loss(vessel_conv, i_parent, mask=mask))/5.0

    return loss
