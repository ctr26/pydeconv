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
from . import simulation, imputation, utils, transfer_matrix

cropped = None
coord_list = None
flat_df = None

def impute(psf_image, psf_centres="auto", normalise=True, centre_crop=True):
    pass

def simulate(dims=[1024,1024],type="guassian",params=[]):
    pass

# def __main__(psf_image, psf_centres, normalise=True, centre_crop=True):
#     """
#     Function for giving in a
#     """
#     eigen_psf = None  # Is a list of coord.shape arrays
#     weighting = None  # Weighting is a function that takes *coords
#     return eigen_psf, weighting


# def getEigenPSF(principle_component):
#     eigen_psfs = pca.components_.reshape((-1, *psf_window))
#     return eigen_psf

# scale = 4.0
# psf_w = 16
# psf_h = 16
# # scale = 1.0
# # int(12/scale)
# static_psf = np.ones((int(12 / scale), int(12 / scale))) / \
#     int(12 / scale)**2  # Boxcar


# def psf_guass(w=psf_w, h=psf_h, sigma=3):
#     # blank_psf = np.zeros((w,h))
#     def gaussian(x, mu, sig):
#         return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

#     xx, yy = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
#     return gaussian(xx, 0, sigma) * gaussian(yy, 0, sigma)


# static_psf = psf_guass(w=psf_w, h=psf_h, sigma=1 / 5)
# plt.imshow(static_psf)






# def __main__(psf_image, psf_centres, normalise=True, centre_crop=True):
#     """
#     Function for giving in a
#     """
#     eigen_psf = None  # Is a list of coord.shape arrays
#     weighting = None  # Weighting is a function that takes *coords
#     return eigen_psf, weighting


# def __main__(psf_image_array,psf_centres,noramlise=True):


#     eigen_psf = None # Is a list of coord.shape arrays
#     weighting = None # Weighting is a function that takes *coords
#     return eigen_psf,weighting

# import numpy as np


# class Psf:

#     dimensionality = ""

#     def set_dimensionality(self, dimensionality):
#         # self.data
#         self.dimensionality = dimensionality

#     def get_dimensionality(self):
#         return self.dimensionality


# import numpy as np


# class Psf:

#     dimensionality = ""

#     def set_dimensionality(self, dimensionality):
#         # self.data
#         self.dimensionality = dimensionality

#     def get_dimensionality(self):
#         return self.dimensionality#


# def variable_gaussian_psf(image,psf_dims,mu,sigma_fun):
#     def psf_fun(r_map):
#         # psf_array = 
#         sigma_map = sigma_fun(r_map)
#         for i,sigma in enumerate(sigma_map):
#             psf_array[coord] = point_spread_function.gaussian(psf_dims,mu,sigma)
#         return psf_array
#     return variable_psf(image,psf_fun)
#     # image_dims = image.shape
#     # grid_coords = np.meshgrid(*[np.linspace(-1, 1, image_dim) for image_dim in image_dims])
#     # r_map = np.add.reduce(np.power(grid_coords,2.0))
#     # sigma_map = sigma_fun(r_map)
#     # r_dist = r_map[coords]
#     # def sigma_scale(r_dist):
#     #     return (r_dist + 0.01) * 3
#     # for i,sigma in enumerate(sigma_map):
#     #     coords = np.unravel_index(i, image_dims.shape)
#     #     psf_array[coords] = point_spread_function.gaussian(psf_dims,mu,sigma)
#     # return 