# Expanded the modified richardson lucy equation to the first two components.
import numpy as np
from functools import wraps

# from scipy.signal.signaltools import deconvolve
from ..utils import xyz_viewer
import matplotlib.pyplot as plt

from pydeconv.simulate import psfs
from pydeconv import optics, utils
from tqdm import tqdm

# from . import early_stopping import EarlyStopping
from . import stopping

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def stopping_decorator(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.check_stopping_conditions(*args, **kwargs):
            raise StopIteration
        return func(self, *args, **kwargs)

    return wrapper


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
    stopping_conditions = []

    def __init__(self, psf, max_iterations=25, early_stopping=None):
        super().__init__(psf=psf)
        self.max_iterations = max_iterations
        if early_stopping is None:
            early_stopping = stopping.NoStopping()
        self.stopping_conditions.append(early_stopping)

    def check_stopping_conditions(self, *args, **kwargs):
        """Check all stopping conditions. Return True if any condition is met."""
        return any(
            condition(*args, **kwargs)
            for condition in self.stopping_conditions
        )

    def step_loop(self, steps, y):
        length = len(steps)
        for i in tqdm(range(length)):
            steps[i] = self._step(y, steps[: i + 1])
        return steps[: i + 1]

    # @stopping_decorator
    def _step(self, y, steps):
        # logger.info(f"Running iteration {iteration}")
        iteration = len(steps) - 1
        return self.step(y=y, steps=steps)

    def step(self, y, steps):
        pass

    def deconvolve(self, image, history=False):
        est_0 = self.est_0(image)
        steps = np.expand_dims(est_0, 0).repeat(self.max_iterations, axis=0)
        # self.steps = steps
        steps = self.step_loop(y=image, steps=steps)
        if history:
            return steps
        return steps[-1]

    # def callback_stopping(self, history, iterations):
    # return self.stopping_conditions(history, iterations),

    def est_0(self, image):
        return image

    def est_grey(self, image):
        return np.ones_like(image) + np.mean(image)

    def est_signal(self, image):
        return self.fwd(image)

    def est_half(self, image):
        return np.ones_like(image) * (image.max() / 2)

