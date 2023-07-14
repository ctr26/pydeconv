from scipy.interpolate import Rbf
import dask.array as da
from dask.diagnostics import ProgressBar
import pandas as pd
import numpy as np
from scipy.signal.filter_design import iirfilter
from scipy.sparse import dok_matrix
from tqdm import tqdm
import dask
import scipy
from sklearn.decomposition import PCA
import sparse
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale

cropped = None
coord_list = None
flat_df = None


def _gaussian_fun(x, mu, sigma):
    """
    0D guassian function
    """
    numerator = -np.power(x - mu, 2.0)
    divisior = np.multiply(2, np.power(sigma, 2.0))
    return np.exp(np.divide(numerator, divisior))


def _gaussian_1D(x, mu, sigma):
    """
    1D guassian function, takes a line of x values with 0-dim mu and sigma
    """
    mu_x = np.full(x.shape, mu)
    sigma_x = np.full(x.shape, sigma)
    return _gaussian_fun(x, mu_x, sigma_x)


def _gaussian_nd(dims, mu, sigma):
    """
    Takes image width and produces a guassian psf
    """
    # Create nd meshgrid
    grid_coords = np.meshgrid(*[np.linspace(-1, 1, dim) for dim in dims])

    gaussian_mesh_list = []
    # Attempt to zip dims my and sigma
    zipped_params = list(zip(mu, sigma, dims, grid_coords))
    for i, zipped_param in enumerate(zipped_params):
        # mu_x = np.full(grid_coords[i].shape,zipped_param[0])
        # sigma_x = np.full(grid_coords[i].shape,zipped_param[1])
        gaussian_mesh_list.append(
            _gaussian_1D(x=zipped_param[3], mu=zipped_param[0], sigma=zipped_param[1])
        )
    # Unormalised result
    normalisation = 1 / (2 * (np.pi) * (np.multiply.reduce(sigma)))
    return normalisation * np.multiply.reduce(gaussian_mesh_list)


def gaussian(dims=[10, 10], mu=[0, 0], sigma=[1, 1]):
    return _gaussian_nd(dims, mu, sigma)


def variable_psf(image, psf_fun):
    """
    Image is only used for finding the extent of the image.
    psf_fun takes in a radius, probably would be better with a normalised coord
    """
    # image_dims = image.shape
    # grid_coords = np.meshgrid(
    #     *[np.linspace(-1, 1, image_dim) for image_dim in image_dims]
    # )
    # r_map = np.add.reduce(np.power(grid_coords, 2.0))

    psf_array = map_of_fun(image, psf_fun)
    # psf_array = psf_fun(r_map)
    return psf_array

    # r_dist = r_map[coords]
    # def sigma_scale(r_dist):
    #     return (r_dist + 0.01) * 3
    # for i,sigma in enumerate(sigma_map):
    #     coords = np.unravel_index(i, image_dims.shape)
    #     psf_array[coords] = point_spread_function.gaussian(psf_dims,mu,sigma)


# def variable_gaussian_psf(image, psf_dims, mu_map, sigma_map):
#     def psf_fun(r_map):
#         psf_array = np.empty([image.shape + psf_dims])
#         sigma_map = sigma_fun(r_map)
#         for i, sigma in enumerate(sigma_map):
#             coords = np.unravel_index(i, image.shape)
#             psf_array[coords, :, :] = gaussian(
#                 psf_dims, mu_map[coords], sigma_map[coords]
#             )
#         return psf_array

#     return variable_psf(image, psf_fun)
#     # image_dims = image.shape
# grid_coords = np.meshgrid(*[np.linspace(-1, 1, image_dim) for image_dim in image_dims])
# r_map = np.add.reduce(np.power(grid_coords,2.0))
# sigma_map = sigma_fun(r_map)
# r_dist = r_map[coords]
# def sigma_scale(r_dist):
#     return (r_dist + 0.01) * 3
# for i,sigma in enumerate(sigma_map):
#     coords = np.unravel_index(i, image_dims.shape)
#     psf_array[coords] = point_spread_function.gaussian(psf_dims,mu,sigma)
# return


def variable_gaussian_psf(image, psf_dims, mu_map, sigma_map):
    fun_map = np.empty(list(image.shape) + list(psf_dims))
    for i, x in enumerate(np.nditer(image)):
        coords = np.unravel_index(i, image.shape)
        # print(x)
        fun_map[coords] = gaussian(psf_dims, mu_map[coords], sigma_map[coords])
    return fun_map


def map_of_fun(x_map, fun):
    fun_map = np.empty(list(x_map.shape) + [x_map.ndim])
    for i, x in enumerate(np.nditer(x_map)):
        coords = np.unravel_index(i, x_map.shape)
        # print(x)
        fun_map[coords] = fun(x)
    return fun_map


def radial_map(image):
    image_dims = image.shape
    grid_coords = np.meshgrid(
        *[np.linspace(-1, 1, image_dim) for image_dim in image_dims]
    )
    return np.add.reduce(np.power(grid_coords, 2.0))



