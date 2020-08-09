import torch
import torch.nn.functional as F
import numpy as np

def null_fn(*args):
    return None

#print_fn = print
print_fn = null_fn

# Output vesselness
def v2vesselness(image, ves, nsample=20, vtype='light', mask=None, percentile=100, is_crosscorr=False, **kwargs):
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
    response = response - response.min()
    if mask is not None:
        response = response * mask
    return response

# Output vesselness over square filters
def v2_sq_vesselness(image, ves, nsample=10, vtype='light', mask=None, percentile=100, is_crosscorr=False, **kwargs):
    response = 0.0
    i_range = []

    v1 = ves*0
    v1[:, 1] = ves[:, 0]
    v1[:, 0] = -ves[:, 1]

    for sv in np.linspace(-2, 2, nsample):
        im = resample_from_flow_2d(image, sv*v1)
        # Get perp profile
        for s in np.linspace(-2, 2, nsample):
            filt = 2*int(abs(s) < 1) - 1
            i_val = resample_from_flow_2d(im, s*ves)
            if is_crosscorr:
                i_range.append(i_val.detach()[:, None])
            # Compute the convolution I * f
            response = response + (i_val * filt)/(nsample**2)

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
    response = response - response.min()
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
    num_samples_template = args.get('num_samples_template', 10)
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


def vessel_loss_2d_sq(output, data, config):
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
    num_samples_template = args.get('num_samples_template', 10)
    l_perlength = args.get('lambda_perlength')
    detach = args.get('detach', True)
    # This parameter is for type of vessel
    vessel_type = config.get('vessel_type', 'light') # should be light, dark or both

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
            vessel_conv = v2_sq_vesselness(image, v2, num_samples_template, vessel_type, is_crosscorr=is_crosscorr)
            loss = loss + l_template * (1 - (mask*vessel_conv).mean())

        # Check for vesselness in followup
        if l_followupv and l_template:
            # we have already calculated the vesselness
            followup_vessel = resample_from_flow_2d(vessel_conv, v1)
            loss = loss + l_followupv * (1 - (mask*followup_vessel).mean()/num_samples_template)

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
