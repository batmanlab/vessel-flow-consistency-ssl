import torch
import torch.nn.functional as F

def nll_loss(output, target):
    return F.nll_loss(output, target)

def L1(i1, i2=0):
    return torch.abs(i1 - i2).mean()

def L2(i1, i2=0):
    return torch.mean((i1 - i2)**2)

# Have a dictionary of keys to loss functions
LOSS_FNs = {
        'L1': L1,
        'L2': L2,
}

def get_grid(image):
    ''' Return an affine grid in range [-1, 1] '''
    B = image.shape[0]
    grid = torch.tensor([[1, 0, 0], [0, 1, 0]])[None]
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


def resample_from_flow_2d(image, flow)
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
    imagenew = F.grid_sample(image+0, gridnew)
    return imagenew


def flow_consistency_2d(parent, child):
    '''
    Given a parent flow, and a child flow, we will output the difference
    Basically, we will output p(x) + c(p(x))
    '''
    B, C, H, W = parent.shape
    size = parent.size()

    xx, yy = get_grid(parent)
    xx, yy = small2largegrid(xx, yy, size))
    # Find parent values
    xp = xx + parent[:, 0]
    yp = yy + parent[:, 1]
    # Given new coordinates, convert them and sample the values of children
    xp, yp = large2smallgrid(xp, yp, size)
    newgrid = xytogrid(xp, yp)
    newchild = F.grid_sample(child, newgrid)
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
    L_loss = LOSS_FNs[args['loss_intensity']]
    # Weights for different parts of total loss
    l_intensity = args.get('lambda_intensity')
    l_consistency = args.get('lambda_consistency')
    l_cosine = args.get('lambda_cosine')
    l_decoder = args.get('lambda_decoder')
    l_length = args.get('lambda_length')

    # Get outputs and inputs
    recon = output['recon']
    vessel = output['vessel']
    image = output['image']

    # Now use the losses given in the config
    if num_dir == 2 and not unc:
        assert config['arch']['args']['out_channels'] == 4, 'Model with 2 directions and no uncertainty'
        # parameters are v1, v2
        loss = 0.0
        v1 = vessel[:, :2]
        v2 = vessel[:, 2:]
        # Intensity consistency loss
        if l_intensity:
            i_parent = resample_from_flow_2d(image, v1)
            i_child = resample_from_flow_2d(image, v2)
            loss = loss + l_intensity * (L_loss(image, i_parent) + L_loss(image, i_child))
        # Flow consistency loss
        if l_consistency:
            loss = loss + l_consistency * L2(flow_consistency_2d(v1, v2))
        # Check for cosine similarity
        if l_cosine:
            loss = loss + l_cosine * F.cosine_similarity(v1, v2).mean()
        # Check for decoder
        if l_decoder:
            loss = loss + l_decoder * L2(image, recon)
        # Check for length of vector
        if l_length:
            v1norm = (v1**2).sum(1) + eps
            v2norm = (v2**2).sum(1) + eps
            loss = loss + l_length * (L2(1./v1norm) + L2(1./v2norm))

    else:
        raise NotImplementedError

    # Return loss
    return loss
