import pydeconv

from pydeconv import optics, utils
from pydeconv import deconvolve
import numpy as np
from tqdm import tqdm

class BionomialSplitting(deconvolve.RichardsonLucy):
    def __init__(self, psf, max_iterations=100):
        super().__init__(
            psf=psf,
            max_iterations=max_iterations,
            early_stopping=BionomialSplittingStopper(),
        )
    
    def stopping

class BinomialSplittingStopper(deconvolve.EarlyStopping):
    def __init__(self):
        super().__init__()

    def __call__(self, steps):
        return False