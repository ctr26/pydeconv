# Expanded the modified richardson lucy equation to the first two components.
import numpy as np
from scipy.signal.signaltools import deconvolve
from ..utils import xyz_viewer
import matplotlib.pyplot as plt

from pydeconv.simulate import optics

class DeconvolveBase:
    def __init__(
        self,
        psf,
        iterations=25,
        early_stopping=None,
        history=False,
    ):
        if history:
            self.history_flag = True
        self.history = np.zeros((iterations, *psf.shape))
        self.iterations = iterations
        if early_stopping:
            self.early_stopping = EarlyStopping()
        self.early_stopping = early_stopping
        self.otf, self.fwd, self.bwd = optics.operators(psf)

    def __call__(self, image):
        return self.deconvolve(image)

    def check_early_stopping(self):
        return self.early_stopping(self.history)

    def step(self, img):
        deconvolved = self.deconvolution_step(img)
        if self.check_early_stopping():
            return self.history[-1]
        return deconvolved

    def deconvolution_step(self, img):
        return img

    # def get_psf(self):
    # return get_optical_operator(self.fwd, self.bwd)

    def deconvolve(self, image):
        for i in range(self.iterations):
            self.history[i] = self.step(image)
        if self.history_flag:
            return self.history
        else:
            return self.history[-1]


class EarlyStopping:
    # Need a smart way of hooking into Deconvolve
    def __init__(self):
        pass
    
    def __call__(self, history):
        return False


class Factory(DeconvolveBase):
    # Do some factory magic to get the right deconvolution method
    pass


# def deconvolve(image, psf_image, method="rl"):
#     pass
