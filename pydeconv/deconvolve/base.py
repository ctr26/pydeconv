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
        iterations=25,
        early_stopping=None,
    ):
        self.iterations = iterations
        self.psf = psf
        self.otf, self.fwd, self.bwd = optics.operators(psf)
        if early_stopping is not None:
            self.early_stopping = early_stopping

    def __call__(self, image):
        return self.deconvolve(image)

    def check_early_stopping(self):
        return self.early_stopping(self.steps)

    def step(self, image, i):
        deconvolved = self.deconvolution_step(image, i)
        if self.check_early_stopping():
            return self.steps[-1]
        return deconvolved

    def deconvolution_step(self, image, i):
        return image

    # def get_psf(self):
    # return get_optical_operator(self.fwd, self.bwd)

    def deconvolve(self, image, history=False):
        self.steps = np.expand_dims(image, 0).repeat(self.iterations, axis=0)
        for i in tqdm(range(self.iterations)):
            self.steps[i] = self.step(image, i)
        if history:
            return self.steps
        return self.steps[-1]

    # def history(self):
    #     return self.history


class Factory(DeconvolveBase):
    # Do some factory magic to get the right deconvolution method
    pass


# def deconvolve(image, psf_image, method="rl"):
#     pass
