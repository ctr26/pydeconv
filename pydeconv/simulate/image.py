from . import  objects, psfs
import numpy as np
from pydeconv import optics, utils

# def bionomial_splitting(img, p=0.5):
#     img_T = np.random.binomial(img, 0.5)
#     img_V = img - img_T
#     return img_T, img_V


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
        max_photons=1e5,
    ):
        obj_name = obj_name
        numPixel = numPixel
        midPos = midPos
        # self.obj = objects.create_object(obj_name, numPixel, midPos)
        self.obj = create_normalised_object(obj_name, numPixel, midPos, max_photons)
        #TODO option for other psfs 
        self.apsf = psfs.jinc_psf(numPixel, midPos, pxSize, lambda0, NA)
        self.psf = optics.apsf2psf(self.apsf)
        self.otf, self.fwd, self.bwd = optics.operators(self.psf)

    def simulate(self):
        return optics.simulate_image(self.obj, self.fwd, noise=True)

    def get_object(self):
        return self.obj

    def get_optical_operators(self):
        return self.psf, self.otf, self.fwd, self.bwd

    def bionomial_splitting(self):
        pass
