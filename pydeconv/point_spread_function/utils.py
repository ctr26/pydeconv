
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

def psf_histogram(cropped):
    sum_cropped = np.sum(cropped, axis=(1, 2, 3))
    plt.hist(sum_cropped)


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