import numpy as np
from scipy import special
from pydeconv import optics, utils


def jinc_apsf(numPixel, midPos, pxSize, lambda0, NA):
    """Assumes isotropic number of pixels (e.g. 256 x 256"""
    lambda0 = lambda0 / pxSize[0]
    abbelimit = lambda0 / NA
    ftradius = numPixel[0] / abbelimit
    scales = ftradius / numPixel[0]

    x = utils.xx(numPixel, numPixel)
    y = utils.yy(numPixel, numPixel)

    r_scaled = np.pi * np.sqrt((x * scales) ** 2 + (y * scales) ** 2)

    apsf = special.jv(1, 2 * r_scaled) / (r_scaled + 1e-12)
    apsf[midPos[0], midPos[1]] = 1.0

    return apsf


def jinc_psf(numPixel, midPos, pxSize, lambda0, NA):
    return optics.apsf2psf(jinc_apsf(numPixel, midPos, pxSize, lambda0, NA))
