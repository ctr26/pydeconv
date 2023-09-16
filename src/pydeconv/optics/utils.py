import numpy as np
from scipy import special, fft, interpolate


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


