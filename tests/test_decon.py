import pydeconv.deconvolve as deconvolve
import pydeconv.point_spread_function as psf

import pytest

from skimage import data,color

# def sigma_scale(r_dist):
#     return (r_dist + 0.01) * 3

# psf_image = psf.gaussian(dims=dims, mu=mu, sigma=sigma)
# image = color.rgb2gray(data.astronaut())

# N_v = np.ma.size(astro);N_v
# N_p = np.ma.size(astro_blur);N_p
# measurement_matrix = matrix(np.zeros((N_p, N_v)))

# for i in np.arange(N_v):
#     coords = np.unravel_index(i, astro.shape)
#     r_dist = r_map[coords]
#     sigma = sigma_scale(r_map[coords])
#     psf_image = psf.gaussian(dims=dims, mu=mu, sigma=sigma)
#     psf_window_volume[i, :, :] = psf_image
#     delta_image = np.zeros_like(astro)
#     delta_image[np.unravel_index(i, astro_shape)] = 1
#     delta_PSF = scipy.ndimage.convolve(delta_image, psf_current)
#     measurement_matrix[i, :] = delta_PSF.flatten()

def test_deconvolve_shape():
    pass
    # assert 

