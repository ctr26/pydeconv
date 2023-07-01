from . import utils
import numpy as np
import numpy as np
from scipy import special, fft, interpolate


class OpticalModel:
    def __init__(self, psf):
        self.psf = psf
        self.otf = psf2otf(psf)

    def fwd(self, x):
        return forward_optical_model(x, self.otf)

    # return np.real(utils.ift2d(utils.ft2d(x) * self.otf))
    def bwd(self, x):
        return backward_optical_model(x, self.otf)
        # return np.real(utils.ift2d(utils.ft2d(x) * np.conj(self.otf)))

# apsf = jinc_psf(numPixel, midPos, pxSize, lambda0, na)

def generate_optical_operators(
    apsf,
    numPixel=128,
    midPos=64,
    pxSize=0.1,
    lambda0=0.5,
    na=0.5,
):
    # TODO nd
    # psf = jincPSF(numPixel, midPos, pxSize, lambda0, NA)
    # obj = obj / np.max(obj) * max_photons
    # Forward and backwards model
    
    psf = utils.abssqr(apsf)
    psf = psf / np.sum(psf) * np.sqrt(np.size(psf))
    otf = utils.ft2d(psf)
    optical_model = OpticalModel(otf)
    # fwd = lambda x: np.real(utils.ift2d(utils.ft2d(x) * otf))
    # bwd = lambda x: np.real(utils.ift2d(utils.ft2d(x) * np.conj(otf)))
    return otf, optical_model.fwd, optical_model.bwd


def psf2otf(apsf):
    psf = utils.abssqr(apsf)
    psf = psf / np.sum(psf) * np.sqrt(np.size(psf))
    otf = utils.ft2d(psf)
    return otf


def forward_optical_model(x, otf):
    return np.real(utils.ift2d(utils.ft2d(x) * otf))


def backward_optical_model(x, otf):
    return np.real(utils.ift2d(utils.ft2d(x) * np.conj(otf)))


def abbe_radius_from_pupil(pupil, midPos):
    arr = np.argwhere(utils.abssqr(pupil / np.max(pupil)) > 1e-3)
    R = np.max(np.sqrt((arr[:, 0] - midPos[0]) ** 2 + (arr[:, 1] - midPos[1]) ** 2))
    return np.ceil(R)


def jinc_psf(numPixel, midPos, pxSize, lambda0, NA):
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
