
from pydeconv.point_spread_function.utils import get_psf_at_coord, get_shapes_from_psf
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
# import point_spread_function
from scipy.sparse import coo_matrix
import numpy as np 
from scipy import matrix



def blur_image_with_H(image,H):
    return H.dot(image.flatten()).reshape(image.shape)

        # This probably goes in point_spread_function?
def generate_H_from_psf_fn(image,psf_function,**kwargs):

    dims = image.shape
    psf_array = psf_function(image=image,**kwargs)

    measurement_matrix = matrix(np.zeros((len(dims), len(psf_array))))

    for i, psf_image in enumerate(psf_array):
        # coords = np.unravel_index(i, dims)
        # psf_image = psf_function(coords)
        # r_dist = r_map[coords]
        # sigma = sigma_scale(r_map[coords])
        # psf_image = psf.gaussian(dims=dims, mu=mu, sigma=sigma)
        # psf_window_volume[i, :, :] = psf_image
        delta_image = np.zeros_like(image)
        delta_image[np.unravel_index(i, image.shape)] = 1
        # This step is slow as it first makes an array and then COOs it
        # Need a method to build COO as we go
        delta_PSF = scipy.ndimage.convolve(delta_image, psf_image)
        measurement_matrix[i, :] = delta_PSF.flatten()
    
    return coo_matrix(measurement_matrix)



def psf_to_H(psf_array, method="dask"):
    if method == "dask":
        return _psf_to_H_dask(psf_array)
    # if method=="linear": return  psf_to_H_dask(psf_array)
    return _psf_to_H_slow(psf_array)


def _psf_to_H_slow(psf_array):
    """ Takes an ND psf array (image_coords,psf_coords)"""
    ### Very slow way of doing this, could dask it.
    # Assumes that there are as many dimensions in the PSF as there are in the image
    # Unsure if this is faulty
    # dims = int(np.divide(len(psf_array.shape), 2))
    # image_shape = psf_array.shape[0:dims]
    # psf_shape = psf_array.shape[dims:]

    image_shape, psf_shape = get_shapes_from_psf(psf_array)

    N_v = np.multiply.reduce(image_shape)
    # Faulty assumption that N_v equals N_p but ok for now
    N_p = np.multiply.reduce(image_shape)

    measurement_matrix = dok_matrix((N_v, N_p))
    for i in tqdm(np.arange(N_v)):
        coords = np.unravel_index(i, image_shape)
        current_psf = psf_array[coords]
        # Get the xy coordinates of the ith pixel in the original image
        delta_image = np.zeros(image_shape)
        # if method == "convolve":
        delta_image[coords] = 1
        delta_PSF = scipy.ndimage.convolve(
            delta_image, current_psf
        )  # Convolve PSF with a image with a single 1 at coord
        # if method == "put":
        # delta_PSF = put_centre(delta_image, current_psf, coords)
        measurement_matrix[i, :] = delta_image.flatten()
    return measurement_matrix


def _psf_to_H_dask(psf_array):
    # dims = int(np.divide(len(psf_array.shape), 2))
    # image_shape = psf_array.shape[0:dims]
    # psf_shape = psf_array.shape[dims:]

    image_shape, psf_shape = get_shapes_from_psf(psf_array)

    N_v = np.multiply.reduce(image_shape)
    # Faulty assumption that N_v equals N_p but ok for now
    N_p = np.multiply.reduce(image_shape)

    dask_list = []
    delayed_psf_array = dask.delayed(psf_array)
    for i in tqdm(np.arange(N_v)):
        # delta_PSF = get_psf_at_coord(psf_array,coord)
        coord = np.unravel_index(i, image_shape)
        delta_dask = dask.delayed(get_psf_at_coord)(delayed_psf_array, coord)
        array_da = da.from_delayed(delta_dask, shape=image_shape, dtype=float)
        # delta_dask_flat = array_da.flatten()
        # sparse_da = delta_dask_flat.map_blocks(sparse.COO)
        # delta_dask_sparse = dask.delayed(dok_matrix)(delta_dask_flat)
        dask_list.append(array_da.flatten())
    # stack = dask.array.concatenate(dask_list,axis=0)

    stack = dask.array.stack(dask_list)
    stack = stack.map_blocks(sparse.COO)
    with ProgressBar():
        out = stack.compute()
    return out


def impute_psf(raw_psf_image_file):
    pass

def impute_psf_from_image(psf_image,order=0):
    pass

def impute_psf_from_image_order_0(psf_image):
    # Average 
    pass

def impute_psf_from_image_order_n(psf_image,order):
    pass

# import point_spread_function
# from scipy.sparse import coo_matrix
# import numpy as np 


# # def generate_H(psf):
# #     H = None
# #     pass
# #     return H


# # def generate_psf(raw_image_file):
# #     psf = None
# #     pass
# #     return psf



# # This probably goes in point_spread_function?
# def generate_H(image,psf_function,**kwargs):
#     dims = image.shape
#     psf_array = psf_function(image=image,**kwargs)
#     for i, psf_image in enumerate(psf_array):
#         # coords = np.unravel_index(i, dims)
#         # psf_image = psf_function(coords)
#         # r_dist = r_map[coords]
#         # sigma = sigma_scale(r_map[coords])
#         # psf_image = psf.gaussian(dims=dims, mu=mu, sigma=sigma)
#         # psf_window_volume[i, :, :] = psf_image
#         delta_image = np.zeros_like(image)
#         delta_image[np.unravel_index(i, image.shape)] = 1
#         # This step is slow as it first makes an array and then COOs it
#         # Need a method to build COO as we go
#         delta_PSF = scipy.ndimage.convolve(delta_image, psf_image)
#         measurement_matrix[i, :] = delta_PSF.flatten()
    
#     return coo_matrix(measurement_matrix)


# # def variable_psf(image,psf_fun):
# #     image_dims = image.shape
# #     grid_coords = np.meshgrid(*[np.linspace(-1, 1, image_dim) for image_dim in image_dims])
# #     r_map = np.add.reduce(np.power(grid_coords,2.0))
# #     psf_array = psf_fun(r_map)
# #     # r_dist = r_map[coords]
# #     # def sigma_scale(r_dist):
# #     #     return (r_dist + 0.01) * 3
# #     # for i,sigma in enumerate(sigma_map):
# #     #     coords = np.unravel_index(i, image_dims.shape)
# #     #     psf_array[coords] = point_spread_function.gaussian(psf_dims,mu,sigma)
# #     return psf_array

# # def generate_H(psf):
# #     H = None
# #     return H

# # def generate_psf(raw_image_file):
# #     psf = None
# #     return psf

# # def impute_psf(raw_image_file):
# #     H = None
# #     return H

