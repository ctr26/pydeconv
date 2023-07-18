# Expanded the modified richardson lucy equation to the first two components.
import numpy as np

# from scipy.signal.signaltools import deconvolve
from ..utils import xyz_viewer
import matplotlib.pyplot as plt

from pydeconv.simulate import psfs
from pydeconv import optics, utils
from tqdm import tqdm


class EarlyStopping:
    # Need a smart way of hooking into Deconvolve
    def __init__(self):
        pass

    def __call__(self, history):
        return False


class DeconvolveBase:
    early_stopping = EarlyStopping()
    steps = None

    def __init__(
        self,
        psf,
        max_iterations=25,
        early_stopping=None,
    ):
        self.max_iterations = max_iterations
        self.psf = psf
        self.otf, self.fwd, self.bwd = optics.operators(psf)
        if early_stopping is not None:
            self.early_stopping = early_stopping

    def __call__(self, image):
        return self.deconvolve(image)

    def check_early_stopping(self):
        return self.early_stopping(self.steps)

    def step(self, image, i):
        return image

    # def get_psf(self):
    # return get_optical_operator(self.fwd, self.bwd)

    def est_0(self, image):
        return image

    def iterative_decovolve(self, image, history=False):
        est_0 = self.est_0(image)
        self.steps = np.expand_dims(est_0, 0).repeat(self.max_iterations + 1, axis=0)
        i = 0
        for i in tqdm(range(self.max_iterations)):
            self.steps[i + 1] = self.step(image, i=i)
            if self.check_early_stopping():
                break
        if history:
            return self.steps
        return self.steps[i]

    def deconvolution_function(self, image):
        return image

    def deconvolve(self, image, history=False):
        if self.max_iterations is not None:
            return self.iterative_decovolve(image, history=history)
        return self.deconvolution_function(image)


class Factory(DeconvolveBase):
    # Do some factory magic to get the right deconvolution method
    pass


# def deconvolve(image, psf_image, method="rl"):
#     pass
