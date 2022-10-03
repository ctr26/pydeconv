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

psf_image = psf.simulate.gaussian(dims=dims, mu=mu, sigma=sigma)
image = color.rgb2gray(data.astronaut())


# def sigma_scale(r):
#     return [r + 0.01, r + 0.01]

# def mu_scale(r):
#     return [0, 0]


def test_psf_varies():
    psf_image_1 = psf.simulate.gaussian(dims=dims, mu=mu, sigma=sigma)
    psf_image_2 = psf.simulate.gaussian(dims=dims, mu=mu, sigma=np.divide(sigma, 2))
    error = (np.abs(psf_image_1 - psf_image_2)).flatten().sum()
    assert error != 0


def test_psf_shape():
    assert list(psf_image.shape) == list(dims)

def test_radial_map():
    r_dist = psf.simulate.radial_map(image)
    assert list(r_dist.shape) == list(image.shape)


def test_variable_guassian():
    r_dist = psf.simulate.radial_map(image)
    sigma_map = psf.simulate.map_of_fun(r_dist, lambda r : [r + 0.01, r + 0.01])
    mu_map = psf.simulate.map_of_fun(r_dist, lambda r : [0, 0])
    variable_psf_image = psf.simulate.variable_gaussian_psf(image, dims, mu_map, sigma_map)
    assert list(variable_psf_image.shape) == list(image.shape) + list(dims)

def test_convolve():
    pass
