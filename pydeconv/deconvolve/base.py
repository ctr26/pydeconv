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
    steps = None

    def __init__(
        self,
        psf,
    ):
        self.psf = psf
        self.otf, self.fwd, self.bwd = optics.operators(psf)

    def __call__(self, image):
        return self.deconvolve(image)

    def deconvolve(self, image, history=False):
        return image


class IterativeDeconvolve(DeconvolveBase):
    
    early_stopping = EarlyStopping()
    def __init__(self, psf, max_iterations=25, early_stopping=None):
        super().__init__(psf=psf)

        self.max_iterations = max_iterations
        if early_stopping is not None:
            self.early_stopping = early_stopping

    def step(self, image, i):
        return image

    def check_early_stopping(self):
        return self.early_stopping(self.steps)

    def est_0(self, image):
        return image

    def est_grey(self, image):
        return np.ones_like(image) + np.mean(image)

    def est_signal(self, image):
        return self.fwd(image)

    def est_half(self, image):
        return np.ones_like(image) * (image.max() / 2)

    def deconvolve(self, image, history=False):
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


class Factory(DeconvolveBase):
    # Do some factory magic to get the right deconvolution method
    pass


# def deconvolve(image, psf_image, method="rl"):
#     pass
