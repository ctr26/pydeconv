from . import utils, objects, optics
import numpy as np


def bionomial_splitting(img, p=0.5):
    img_T = np.random.binomial(img, 0.5)
    img_V = img - img_T
    return img_T, img_V


def simulate_image(obj, fwd, noise=True):
    img = fwd(obj)
    # Apply shot noise
    if noise:
        return np.random.poisson(img).astype("int32")
    return img


def create_normalised_object(obj_name, numPixel, midPos, max_photons=2**16):
    obj = objects.create_object(obj_name, numPixel, midPos)
    obj = obj / np.max(obj) * max_photons
    return obj


class SimulateImagePoisson:
    def __init__(
        self,
        obj_name="spokes",
        numPixel=128,
        midPos=64,
        pxSize=0.1,
        lambda0=0.5,
        NA=0.5,
    ):
        obj_name = obj_name
        numPixel = numPixel
        midPos = midPos
        self.obj = objects.create_object(obj_name, numPixel, midPos)
        self.psf = optics.jinc_psf(numPixel, midPos, pxSize, lambda0, NA)
        self.otf, self.fwd, self.bwd = optics.generate_optical_operators(
            numPixel, midPos, pxSize, lambda0, NA
        )

        def simulate(self):
            return simulate_image(self.obj, self.fwd, noise=True)

        def get_object(self):
            return self.obj

        def get_optical_operators(self):
            return self.psf, self.otf, self.fwd, self.bwd

        def bionomial_splitting(self):
            pass
