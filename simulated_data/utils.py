import numpy as np

def image_sample(img, x, y, cval=0):
    assert x.shape == y.shape, 'Coordinates need to have the same shape'
    # sample img given fractional x and y
    H, W = img.shape[:2]
    if len(img.shape) == 3:
        C = img.shape[2]
    else:
        C = 1
    outimg = np.zeros((*x.shape, C)) + cval
    # Get valid coordinates
    valididx = (0 <= x)*(x < W-1)*(0 <= y)*(y < H-1)
    nullidx = ~valididx
    # for all valid coordinates, get the values of intensity
    xf = np.floor(x[valididx]).astype(int)
    xc = xf + 1
    yf = np.floor(y[valididx]).astype(int)
    yc = yf + 1
    _x, _y = x[valididx], y[valididx]
    # find values
    img_sampled =     img[yf, xf]*(yc - _y)*(xc - _x) \
                    + img[yf, xc]*(yc - _y)*(_x - xf) \
                    + img[yc, xf]*(_y - yf)*(xc - _x) \
                    + img[yc, xc]*(_y - yf)*(_x - xf)
    if C == 1:
        outimg[valididx, 0] = img_sampled.squeeze()
    else:
        outimg[valididx] = img_sampled
    return outimg.squeeze()



