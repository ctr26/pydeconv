# import pydeconv
import pydeconv.point_spread_function as psf
from skimage import data,color
import matplotlib.pyplot as plt
import numpy as np

from skimage.transform import rescale, resize, downscale_local_mean


def test_2D_variable_psf():
    '''
    Cant think of a better way of doing this
    '''
    # from .notebooks import *
