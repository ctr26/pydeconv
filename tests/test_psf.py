import pydeconv.deconvolve as deconvolve
import pydeconv.point_spread_function as psf
import scipy.
import pytest

from skimage import data,color



# get astronaut from skimage.data in grayscale
dims = [10,10]
mu = [0,0]
sigma = [0.2,0.2]

psf = psf.gaussian(dims=dims, mu=mu, sigma=sigma)
image = color.rgb2gray(data.astronaut())


def test_psf_shape():
    assert list(psf.shape) == dims

def test_convolve():
    pass

