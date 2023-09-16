import numpy as np
from scipy import special, fft, interpolate

def xx(numPixel, simSize):
    x = np.linspace(-simSize[0] / 2, simSize[0] / 2, numPixel[0])
    y = np.linspace(-simSize[1] / 2, simSize[1] / 2, numPixel[1])
    xx = np.meshgrid(x, y)[0]
    return xx


def yy(numPixel, simSize):
    x = np.linspace(-simSize[0] / 2, simSize[0] / 2, numPixel[0])
    y = np.linspace(-simSize[1] / 2, simSize[1] / 2, numPixel[1])
    yy = np.meshgrid(x, y)[1]
    return yy


def rr(numPixel, simSize):
    x = xx(numPixel, simSize)
    y = yy(numPixel, simSize)
    r = np.sqrt(x**2 + y**2)
    return r


def phi(numPixel):
    x = xx(numPixel, numPixel)
    y = yy(numPixel, numPixel)

    phi = np.arctan2(y, x)
    return phi



def ft2d(data_in):
    mySize = np.ndim(data_in)
    myAxes = np.linspace(mySize - 1 - 1, mySize - 1, 2, dtype="int64")
    temp = fft.ifftshift(data_in, axes=myAxes)
    temp = fft.fftn(temp, axes=myAxes, norm="ortho")
    data_out = fft.fftshift(temp, axes=myAxes)

    return data_out


def ift2d(data_in):
    mySize = np.ndim(data_in)
    myAxes = np.linspace(mySize - 1 - 1, mySize - 1, 2, dtype="int64")
    temp = fft.ifftshift(data_in, axes=myAxes)
    temp = fft.ifftn(temp, axes=myAxes, norm="ortho")
    data_out = fft.fftshift(temp, axes=myAxes)
    return data_out


def abssqr(x):
    return np.abs(x) ** 2


def cat(data_in):
    return np.stack(data_in, axis=0)


def radialmean(data, center, nbins):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    r = r.astype(np.int32)

    tbin = np.bincount(r.ravel(), data.ravel())
    if True:
        X = np.arange(0, np.shape(tbin)[0])
        Xnew = np.arange(0, nbins)
        f = interpolate.interp1d(X, tbin, kind="cubic")
        tbin = f(Xnew)

    nr = np.bincount(r.ravel())
    if True:
        f = interpolate.interp1d(X, nr, kind="cubic")
        nr = f(Xnew)

    radialprofile = tbin / nr

    return radialprofile


