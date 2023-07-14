# Expanded the modified richardson lucy equation to the first two components.
import numpy as np
from scipy.signal.signaltools import deconvolve
from ..utils import xyz_viewer
import matplotlib.pyplot as plt

from . import richardson_lucy


class Deconvolve:
    def __init__(self, psf, method="rl", iterations=25, early_stopping=None):
        # self.image = image
        self.psf = psf
        self.method = method
        self.iterations = iterations
        self.early_stopping = early_stopping

    def __call__(self, image):
        return self.deconvolve(image)

    def check_early_stopping(self):
        self.early_stopping()
        pass

    def get_psf(self):
        return self.psf

    def deconvolve(self, image):
        pass


class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta

    def __call__(self, history):
        pass


class RichardsonLucy(Deconvolve):
    def __init__(self, image, psf, iterations=25, early_stopping=None):
        super().__init__(
            image, psf, iterations=iterations, early_stopping=early_stopping
        )
        self.method = "rl"

    def __call__(self):
        return richardson_lucy.richardson_lucy(
            self.image, self.psf, self.iterations, self.early_stopping
        )


def deconvolve(image, psf_image, method="rl"):
    pass


def richarson_lucy_step(est,img,fwd,bwd):
    convEst = fwd(est)
    ratio = img / (convEst + 1e-12)
    convRatio = bwd(ratio)
    convRatio = convRatio / bwd(np.ones_like(img))
    return est * convRatio
