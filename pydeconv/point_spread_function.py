from scipy.interpolate import Rbf
import dask.array as da
from dask.diagnostics import ProgressBar
import pandas as pd
import numpy as np

cropped = None
coord_list = None
flat_df = None

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

    psf_array = map_of_fun(image,psf_fun)
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
    fun_map = np.empty(list(image.shape)+list(psf_dims))
    for i, x in enumerate(np.nditer(image)):
        coords = np.unravel_index(i, image.shape)
        # print(x)
        fun_map[coords] = gaussian(psf_dims,mu_map[coords],sigma_map[coords])
    return fun_map

def map_of_fun(x_map, fun):
    fun_map = np.empty(list(x_map.shape)+[x_map.ndim])
    for i, x in enumerate(np.nditer(x_map)):
        coords = np.unravel_index(i, x_map.shape)
        # print(x)
        fun_map[coords] = fun(x)
    return fun_map

def radial_map(image):
    image_dims = image.shape
    grid_coords = np.meshgrid(*[np.linspace(-1, 1, image_dim) for image_dim in image_dims])
    return np.add.reduce(np.power(grid_coords,2.0))

def df_to_eigen_psf(df):
    pca_df = pd.DataFrame(pca.transform(df), index=df.index)
    eigen_psfs_list = []
    mu = df.mean(axis=0)
    eigen_psfs = pca.components_.reshape((-1, *psf_window))

    eigen_psfs_list.append(mu)
    eigen_psfs_list.append(eigen_psfs)

    # pc_weights_list = []
    # pc_weights_list["mu"] = 1
    # pc_weights.append(pca_df)

    # plt.scatter(pca_df[0], pca_df[1])
    # plt.show()
    # pca.fit_transform(cropped)
    # n_components = 5
    # include average
    # eigen_psfs = pca.components_.reshape((-1, *psf_window))
    # cum_sum_exp_var = np.cumsum(pca.explained_variance_ratio_)
    # accuracy = cum_sum_exp_var[n_components]
    return eigen_psfs_list


def get_pc_weighting_fun_3D(pca_df, function="cubic"):

    melt_df_slim = pd.melt(
        pca_df[[pc_component]],
        var_name="PC",
        value_name="Weight",
        ignore_index=False,
    ).reset_index()

    rbfi = Rbf(
        melt_df["PC"],
        melt_df["z"],
        melt_df["y"],
        melt_df["x"],
        melt_df["Weight"],
        function="cubic",
    )
    weighting_function = rbfi
    # f = interp2d(x_list, y_list, pc_weights[0])

    return weighting_function


def get_pc_weighting_fun_3D(pca_df, pc_component, basis_function="cubic", chunks=5):
    x_list = pca_df.index.get_level_values("x")
    y_list = pca_df.index.get_level_values("y")
    z_list = pca_df.index.get_level_values("z")

    grid_z_psf, grid_y_psf, grid_x_psf = np.mgrid[
        min(z_range_psf) : max(z_range_psf),
        min(y_range_psf) : max(y_range_psf),
        min(x_range_psf) : max(x_range_psf),
    ]

    rbi_fun = get_pc_weighting_fun_3D(pca_df, function="cubic")

    dask_xyzp = da.from_array((grid_z, grid_y, grid_x), chunks=chunks)

    with ProgressBar():
        f = da.map_blocks(rbfi, *dask_xyzp)
    # g = client.persist(f)
    pc_component_weight_map = f.compute()

    return weighting_function


def interpolate_pc_weighting(principle_component, coords):
    weighting = None
    return weighting


def getPSFdf():
    # flat = pd.DataFrame(cropped.reshape(cropped.shape[0], -1)).dropna(0).set_index(index)
    return (
        pd.DataFrame(cropped.reshape(cropped.shape[0], -1)).dropna(0).set_index(index)
    )


# def getPSFdf():
#     # flat = pd.DataFrame(cropped.reshape(cropped.shape[0], -1)).dropna(0).set_index(index)
#     return flat


def __main__(psf_image, psf_centres, noramlise=True, centre_crop=True):
    """
    Function for giving in a
    """
    eigen_psf = None  # Is a list of coord.shape arrays
    weighting = None  # Weighting is a function that takes *coords
    return eigen_psf, weighting


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