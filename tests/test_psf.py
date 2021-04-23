import pydeconv.deconvolve as deconvolve
import pydeconv.point_spread_function as psf
import numpy as np
import pytest

from skimage import data, color


# get astronaut from skimage.data in grayscale
# class test_psf():
        
dims = [10, 10]
mu = [0, 0]
sigma = [0.2, 0.2]

psf_image = psf.gaussian(dims=dims, mu=mu, sigma=sigma)
image = color.rgb2gray(data.astronaut())


def test_psf_varies():
    psf_image_1 = psf.gaussian(dims=dims, mu=mu, sigma=sigma)
    psf_image_2 = psf.gaussian(dims=dims, mu=mu, sigma=np.divide(sigma,2))
    error = (psf_image_1 - psf_image_2).flatten().sum()
    assert error != 0


def test_psf_shape():
    assert list(psf_image.shape) == dims


def test_convolve():
    pass
