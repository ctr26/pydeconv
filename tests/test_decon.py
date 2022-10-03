import pydeconv.deconvolve as deconvolve
import pydeconv.point_spread_function as psf
import pydeconv.utils as utils
from pydeconv.point_spread_function.transfer_matrix import impute_psf_from_image

from os.path import expanduser

import pytest

from skimage import data,color
import pims


real_3D_image_file = "~/+projects/2019_jrl/2019_jRL_impute/data/CraigDeconvolutionData/8bit_bf.tif"
real_3D_beads_file = "~/+projects/2019_jrl/2019_jRL_impute/data/CraigDeconvolutionData/2020-09-04 - calibration/beads/200904_16.50.57_Step_Size_-0.4_Wavelength_DAPI 452-45_500nm_TetBeads/MMStack.ome.tif"

real_3D_image_file_full = expanduser(real_3D_image_file)
real_3D_beads_file_full = expanduser(real_3D_beads_file)

real_3D_psf_window = (140, 40, 40)  # (z,y,x)
real_3D_psf_window = [60, 20, 20]

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

# def airy_get_psf():
    
@pytest.mark.parametrize("dimensions", ["2D", "3D"])
@pytest.mark.parametrize("psf", ["variable", "homogenous"])
@pytest.mark.parametrize("simulation", ["real", "sim"])
def test_deconvolve(dimensions,psf,simulation):
    image,psf_image = get_image(dimensions,simulation)
    if(psf=="variable"):
        psf_order = -1
    if(psf=="homogenous"):
        psf_order = 0

    H = impute_psf_from_image(psf_image,order=0)
    deconvolve.deconvolve(image,H)
    pass

def get_image(dimensions,simulation):
    # if dimensions = "2D":
        
    if (dimensions == "3D") & (simulation == "sim"):
        image_file = real_3D_image_file_full
        beads_file = real_3D_beads_file
        psf_window = real_3D_psf_window

    psf_image = pims.open(beads_file)
    image = pims.open(image_file)

    if (dimensions == "2D") & (simulation == "sim"):
        middle_plane = round(len(image)/2)
        psf_2D = psf_image[middle_plane]
        image_2D = image[middle_plane]

        psf_image = psf_2D
        image = image_2D

# # %% Lazy load images

# psf = pims.open(beads_file)
# image = pims.open(image_file)
#     if simulation = "variable":

#     if simulation = "variable":
        
