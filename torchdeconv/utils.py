import numpy as np
import matplotlib.pyplot as plt
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
from scipy.linalg import circulant
import scipy
# import sparse
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
# import point_spread_function
from scipy.sparse import coo_matrix
import numpy as np 
from scipy import matrix



def scaleImage(image, dtype=np.uint8):
    image = np.array(image)
    scaled = image / image.max()
    scaled_255 = scaled * (np.iinfo(dtype).max)

    scaled_255_8bit = scaled_255.astype(dtype)
    # output = scaled_255_8bit
    return scaled_255_8bit


def cropND(img, centre, window):

    centre = np.array(centre)
    centre_dist = np.divide(window, 2)
    shape = img.shape
    crops = []
    # for i, dim in enumerate(shape):
    #     l = (centre[-(i + 1)] - np.floor(centre_dist[-(i + 1)])).astype(int)
    #     r = (centre[-(i + 1)] + np.ceil(centre_dist[-(i + 1)])).astype(int)

    x_l = (centre[2] - np.floor(centre_dist[2])).astype(int)
    x_r = (centre[2] + np.ceil(centre_dist[2])).astype(int)

    y_l = (centre[1] - np.floor(centre_dist[1])).astype(int)
    y_r = (centre[1] + np.ceil(centre_dist[1])).astype(int)

    z_l = (centre[0] - np.floor(centre_dist[0])).astype(int)
    z_r = (centre[0] + np.ceil(centre_dist[0])).astype(int)
    # try:
    #     return util.crop(img,((z_l,z_r),(y_l,y_r),(x_l,x_r)))
    # except :
    #     return

    return img[z_l:z_r, y_l:y_r, x_l:x_r]

def cropNDv(img, centres, window=[120, 40, 40]):
    cropped_list = np.full((len(centres), *window), np.nan)
    i = 0
    centres_list = []
    for centre in centres:
        try:
            cropped_list[i , :, :, :] = cropND(img, centre, window=window)
            centres_list.append(centres[i])
            i += 1
        except:
            None
    return cropped_list, centres_list

def xyz_viewer(im):
    fig, ax = plt.subplots(1, 3, figsize=(15, 15))
    ax[0].imshow(np.max(im, axis=0))
    ax[1].imshow(np.max(im, axis=1))
    ax[2].imshow(np.max(im, axis=2))
    return fig





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


def get_shapes_from_psf(psf_array):
    dims = int(np.divide(len(psf_array.shape), 2))
    image_shape = psf_array.shape[0:dims]
    psf_shape = psf_array.shape[dims:]
    return image_shape, psf_shape



def get_shapes_from_psf(psf_array):
    dims = int(np.divide(len(psf_array.shape), 2))
    image_shape = psf_array.shape[0:dims]
    psf_shape = psf_array.shape[dims:]
    return image_shape, psf_shape


def get_psf_at_coord(psf_array, coord):
    # dims = int(np.divide(len(psf_array.shape), 2))
    # image_shape = psf_array.shape[0:dims]
    # psf_shape = psf_array.shape[dims:]

    image_shape, psf_shape = get_shapes_from_psf(psf_array)

    # coord = np.unravel_index(i, image_shape)
    current_psf = psf_array[coord]
    # Get the xy coordinates of the ith pixel in the original image
    delta_image = np.zeros(image_shape)
    # if method == "convolve":
    delta_image[coord] = 1
    delta_PSF = scipy.ndimage.convolve(
        delta_image, current_psf
    )  # Convolve PSF with a image with a single 1 at coord
    # if method == "put":
    #     delta_PSF = put_centre(delta_image, current_psf, coord)
    return delta_PSF


def put_centre(array, inset, coord, mode="clip"):
    put_coord = coord - np.floor(np.divide(inset.shape, 2)).astype(int)
    np.put(array, put_coord, inset, mode=mode)
    return array


def show_psf_grid(variable_psf_image, rows=5, cols=5):
    fig, axs = plt.subplots(rows, cols)
    grid_shape = variable_psf_image[:, :, 0, 0].shape
    segments = np.divide(grid_shape, [rows + 1, cols + 1])
    # for i,ax in enumerate(axs):
    for i, row in enumerate(axs):
        for j, ax in enumerate(row):
            axs[i, j].imshow(
                variable_psf_image[
                    int(segments[0] * (i + 1)), int(segments[1] * (j + 1)), :, :
                ]
            )
    return fig

def make_circulant_from_cropped_psf(psf, in_shape, out_shape):
    padding = np.rint(np.divide((np.subtract(out_shape, in_shape)), 2)).astype(int)
    padded_psf = np.pad(
        psf.reshape(in_shape), pad_width=padding, mode="constant", constant_values=0
    )
    centre_coord = np.ravel_multi_index(
        np.divide(padded_psf.shape, 2).astype(int), dims=padded_psf.shape
    )
    rolled_psf = np.roll(padded_psf.flatten(), centre_coord)
    C = scipy.linalg.circulant(rolled_psf)
    return C, rolled_psf