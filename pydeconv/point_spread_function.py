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


def df_to_eigen_psf(df, psf_window, n_components=0.95):
    pca = PCA(n_components=n_components)
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


def get_df_from_var_psf(cropped,coord_list):
    index = pd.MultiIndex.from_arrays(
        np.array(coord_list).transpose(), names=("z", "y", "x")
    )
    flat =  pd.DataFrame(cropped.reshape(cropped.shape[0], -1)).dropna(0).set_index(index)
    return flat


def remove_outliers_sigma(psf_df, sigma=2):
    psf_df
    iirfilter
    low_thres = (psf_df.sum(axis=1).mean()) - sigma * (psf_df.sum(axis=1).std())
    up_thres = (psf_df.sum(axis=1).mean()) + sigma * (psf_df.sum(axis=1).std())
    return psf_df[(psf_df.sum(axis=1) > low_thres) & (psf_df.sum(axis=1) < up_thres)]

def get_outliers_sigma(psf_df, sigma=2):
    psf_df
    low_thres = (psf_df.sum(axis=1).mean()) - sigma * (psf_df.sum(axis=1).std())
    up_thres = (psf_df.sum(axis=1).mean()) + sigma * (psf_df.sum(axis=1).std())
    return psf_df[(psf_df.sum(axis=1) > low_thres) & (psf_df.sum(axis=1) < up_thres)]


def scale_df(df, scale=minmax_scale):
    return df.apply(scale, axis=1, result_type="broadcast")


def normalise_df(df):
    return df.apply(lambda x: x / x.sum(), axis=1)


from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity


def remove_outliers_by_model(df, psf_window):
    outlier_df = df.copy()
    mu = df.mean(axis=0)
    outlier_df["mse"] = df.apply(lambda x: mean_squared_error(x, mu), axis=1)
    outlier_df["ssim"] = df.apply(
        lambda x: structural_similarity(
            x.to_numpy().reshape(psf_window), mu.to_numpy().reshape(psf_window)
        ),
        axis=1,
    )


def get_mse_df(df):
    mu = df.mean(axis=0)
    return df.apply(lambda x: mean_squared_error(x, mu), axis=1)


def get_ssim_df(df, psf_window):
    mu = df.mean(axis=0)
    return df.apply(
        lambda x: structural_similarity(
            x.to_numpy().reshape(psf_window), mu.to_numpy().reshape(psf_window)
        ),
        axis=1,
    )


from sklearn.neighbors import LocalOutlierFactor


def get_inliers_from_forest(df):
    # outlier_df_short = outlier_df.copy()
    clf = LocalOutlierFactor(contamination=0.05).fit(df)
    inliers = clf.fit_predict(df)
    df = df[inliers == 1]
    # print(f"Dropped {np.sum(outliers==1)} outliers")
    return inliers == 1

def get_outliers_from_forest(df):
    return ~(get_inliers_from_forest(df))

def get_image_from_df(df, i, psf_window):
    return df.iloc[i, :].to_numpy().reshape(psf_window)


from scipy.interpolate import interp2d


def get_pc_weighting_fun_2D(pca_df, pc_component, function="cubic"):
    x_list = pca_df.index.get_level_values("x")
    y_list = pca_df.index.get_level_values("y")
    z_list = pca_df.index.get_level_values("z")

    return interp2d(x_list, y_list, pca_df[[pc_component]])


def get_xyz_coord(pca_df, psf_shape):

    x_list = pca_df.index.get_level_values("x")
    y_list = pca_df.index.get_level_values("y")
    z_list = pca_df.index.get_level_values("z")
    # TODO decide on coord order z = 0 or z -1
    z_coords = np.linspace(min(z_list), max(z_list), num=psf_shape[2])
    y_coords = np.linspace(min(y_list), max(y_list), num=psf_shape[1])
    x_coords = np.linspace(min(x_list), max(x_list), num=psf_shape[0])
    return x_coords, y_coords, z_coords


from scipy.interpolate import Rbf
import dask.array as da
from dask.diagnostics import ProgressBar


def get_coord_lists_from_pca_df(pca_df):
    x_list = pca_df.index.get_level_values("x")
    y_list = pca_df.index.get_level_values("y")
    z_list = pca_df.index.get_level_values("z")
    return x_list, y_list, z_list


def get_grid_tuple_from_pca_df(pca_df, spacings=[10, 3, 100, 100]):
    n_components, z_space, y_space, x_space = spacings
    # grid_pc, grid_z, grid_y, grid_x
    x_list, y_list, z_list = get_coord_lists_from_pca_df(pca_df)
    grid_tuple = np.mgrid[
        0:n_components,
        min(z_list) : max(z_list) : z_space * 1j,
        min(y_list) : max(y_list) : y_space * 1j,
        min(x_list) : max(x_list) : x_space * 1j,

    ]
    return grid_tuple

def get_grid_tuple_from_pca_df_3D(pca_df, spacings=[3, 100, 100]):
    z_space, y_space, x_space = spacings
    # grid_pc, grid_z, grid_y, grid_x
    x_list, y_list, z_list = get_coord_lists_from_pca_df(pca_df)
    grid_tuple = np.mgrid[
        min(z_list) : max(z_list) : z_space * 1j,
        min(y_list) : max(y_list) : y_space * 1j,
        min(x_list) : max(x_list) : x_space * 1j,

    ]
    return grid_tuple

def get_dask_grid_tuple_from_pca_df(pca_df, spacings, chunks=5):
    grid_tuple = get_grid_tuple_from_pca_df(pca_df, spacings)
    return da.from_array(grid_tuple, chunks=5)

def get_dask_grid_tuple_from_pca_df_3D(pca_df, spacings, chunks=5):
    grid_tuple = get_grid_tuple_from_pca_df_3D(pca_df, spacings)
    return da.from_array(grid_tuple, chunks=5)

# TODO make ND
def get_cropping_grid(cropping):
    image_min, image_max = cropping

    # image_centre = np.divide(image_shape, 2)
    # image_min = image_centre - np.divide(np.ones_like(image_centre) * cropped_size, 2)
    # image_max = image_centre + np.divide(np.ones_like(image_centre) * cropped_size, 2)

    # = psf_window
    # image_max_psf = cropped_size - psf_window

    x_range_psf = np.floor(np.arange(image_min[2], image_max[2])).astype(int)
    y_range_psf = np.floor(np.arange(image_min[1], image_max[1])).astype(int)
    z_range_psf = np.floor(np.arange(image_min[0], image_max[0])).astype(int)
    # grid_z, grid_y, grid_x = np.meshgrid(x_range,y_range, z_range)

    grid_psf = np.mgrid[
        min(z_range_psf) : max(z_range_psf),
        min(y_range_psf) : max(y_range_psf),
        min(x_range_psf) : max(x_range_psf)
    ]

    return grid_psf


# def get_grid(image_np_shape):
#     x_range_psf = np.floor(np.arange(image_min_psf[0], image_max_psf[0])).astype(int)
#     y_range_psf = np.floor(np.arange(image_min_psf[1], image_max_psf[1])).astype(int)
#     z_range_psf = np.floor(np.arange(image_min_psf[2], image_max_psf[2])).astype(int)
#     # grid_z, grid_y, grid_x = np.meshgrid(x_range,y_range, z_range)

#     return


def map_dask_grid_to_rbf(dask_xyzp, rbfi):
    with ProgressBar():
        f = da.map_blocks(
            rbfi,
            # dask_xyzp[0, :, :, :],
            # dask_xyzp[1, :, :, :],
            # dask_xyzp[2, :, :, :],
            # dask_xyzp[3, :, :, :],
            *dask_xyzp
        )
        # g = client.persist(f)
        g = f.compute()
        # g = progress(client.persist(f))
    return g


def dask_compute_weightings(grid_tuple, rbfi):
    dask_xyzp = da.from_array(grid_tuple, chunks=5)

    # Slow
    # from dask.distributed import progress

    with ProgressBar():
        f = da.map_blocks(
            rbfi,
            dask_xyzp[0, :, :, :],
            dask_xyzp[1, :, :, :],
            dask_xyzp[2, :, :, :],
            dask_xyzp[3, :, :, :],
        )
        # g = client.persist(f)
        g = f.compute()


def get_pc_weighting_fun_3D(pca_df, pc_component, function="cubic"):

    melt_df = pd.melt(
        pca_df[[pc_component]],
        var_name="PC",
        value_name="Weight",
        ignore_index=False,
    ).reset_index()

    rbfi = Rbf(
        # melt_df["PC"],
        melt_df["z"],
        melt_df["y"],
        melt_df["x"],
        melt_df["Weight"],
        function=function,
    )
    weighting_function = rbfi
    # f = interp2d(x_list, y_list, pc_weights[0])

    return weighting_function


def reconstruct_psf_from_pca(pca_df, mu, pca, nComp, psf_window):
    Xhat = np.dot(pca_df.to_numpy()[:, :nComp], pca.components_[:nComp, :])
    Xhat += mu
    reconstructed_psfs = Xhat.reshape(-1, *psf_window)
    return reconstructed_psfs


def plot_reconstruction_scores(
    pca_df, mu, pca, psf_window, example_psf_idx, psf_at_idx, components=50
):
    # components = 50
    mse_list = []
    ssim_list = []

    for nComp in range(components):

        reconstructed_psfs = reconstruct_psf_from_pca(
            pca_df, mu, pca, nComp, psf_window
        )

        mse = mean_squared_error(
            reconstructed_psfs[example_psf_idx, :, :, :].flatten(), psf_at_idx.flatten()
        )
        ssim = structural_similarity(
            reconstructed_psfs[example_psf_idx, :, :, :].flatten(), psf_at_idx.flatten()
        )

        mse_list.append(mse)
        ssim_list.append(ssim)

    fig, ax1 = plt.subplots()

    ax1.plot(mse_list)
    ax2 = ax1.twinx()
    ax2.plot(ssim_list)
    return fig


import seaborn as sns


def plot_pca_weights(pca_df):

    # xy_coords = np.array(coord_list)[:, [0, 2]]
    # x_list = xy_coords[:, 0]
    # y_list = xy_coords[:, 1]

    x_list = pca_df.index.get_level_values("x")
    y_list = pca_df.index.get_level_values("y")
    z_list = pca_df.index.get_level_values("z")

    # pc_weights = np.sqrt((pca_df.iloc[:, 0:] ** 2).sum(axis=1))
    # pc_weights = pca_df[0]
    pc_weights = pca_df

    # pc_weights[pc_weights>500] = np.nan

    return sns.scatterplot(x=x_list, y=y_list, hue=pc_weights[0])
    # plt.plot(xy_coords[0,:],xy_coords[1,:],c=pc_weights)


interpolate_pca_rbf = get_pc_weighting_fun_3D


def get_pc_weighting_map_3D(
    pca_df, image_grid, pc_component, basis_function="cubic", chunks=5
):
    # x_list = pca_df.index.get_level_values("x")
    # y_list = pca_df.index.get_level_values("y")
    # z_list = pca_df.index.get_level_values("z")

    # grid_z_psf, grid_y_psf, grid_x_psf = np.mgrid[
    #     min(z_range_psf) : max(z_range_psf),
    #     min(y_range_psf) : max(y_range_psf),
    #     min(x_range_psf) : max(x_range_psf),
    # ]

    grid_z, grid_y, grid_x = image_grid
    rbi_fun = get_pc_weighting_fun_3D(pca_df, pc_component, function=basis_function)

    dask_xyzp = da.from_array((grid_z, grid_y, grid_x), chunks=chunks)

    with ProgressBar():
        f = da.map_blocks(rbi_fun, *dask_xyzp)
    # g = client.persist(f)
    pc_component_weight_map = f.compute()

    return pc_component_weight_map


def interpolate_pc_weighting(principle_component, coords):
    weighting = None
    return weighting


def getPSFdf(index):
    # flat = pd.DataFrame(cropped.reshape(cropped.shape[0], -1)).dropna(0).set_index(index)
    return (
        pd.DataFrame(cropped.reshape(cropped.shape[0], -1)).dropna(0).set_index(index)
    )


# def getPSFdf():
#     # flat = pd.DataFrame(cropped.reshape(cropped.shape[0], -1)).dropna(0).set_index(index)
#     return flat
from scipy.signal import find_peaks


def outliers_by_single_peak(df, psf_window):
    # TODO add option for peaks in x y and z
    # TODO make it for inliers
    outliers_peaks = df.apply(
        lambda x: len(
            find_peaks(np.max(x.to_numpy().reshape(psf_window), axis=(0, 1)))[0]
        )
        > 1,
        axis=1,
    )
    outliers_peaks |= df.apply(
        lambda x: len(
            find_peaks(np.max(x.to_numpy().reshape(psf_window), axis=(0, 2)))[0]
        )
        > 1,
        axis=1,
    )
    return outliers_peaks == 1


def inliers_by_single_peak(df, psf_window):
    return ~(outliers_by_single_peak(df, psf_window))


def psf_to_H(psf_array, method="dask"):
    if method == "dask":
        return psf_to_H_dask(psf_array)
    # if method=="linear": return  psf_to_H_dask(psf_array)
    return psf_to_H_slow(psf_array)


def psf_df_pca(df, pca=PCA(n_components=0.99)):
    pca = pca.fit(df)
    pca_df = pd.DataFrame(pca.transform(df), index=df.index)
    return pca_df, pca


def eigen_psfs_from_pca(pca, psf_window):
    eigen_psfs = pca.components_.reshape((-1, *psf_window))
    return eigen_psfs


def accuracy_from_pca(pca):
    cum_sum_exp_var = np.cumsum(pca.explained_variance_ratio_)
    # accuracy = cum_sum_exp_var[n_components]
    return cum_sum_exp_var


def psf_to_H_slow(psf_array):
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


def psf_to_H_dask(psf_array):
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


from scipy import ndimage


def getPSFcoords(image, window, sigma=6):
    # scaled = image / image.max()

    # plt.imshow(eq_scaled_255_8bit.max(axis=0))

    img = image

    # Get local maximum values of desired neighborhood
    # I'll be looking in a 5x5x5 area
    print("Max filtering")
    img_max = ndimage.maximum_filter(img, size=np.divide(window, 2))

    # Threshold the image to find locations of interest
    # I'm assuming 6 standard deviations above the mean for the threshold
    print("Thresholding")
    img_thresh = img_max.mean() + img_max.std() * sigma

    # Since we're looking for maxima find areas greater than img_thresh
    print("Labelling")
    labels, num_labels = ndimage.label(img_max > img_thresh)

    # Get the positions of the maxima
    print("Getting coords from centre of mass")
    coords = ndimage.measurements.center_of_mass(
        img, labels=labels, index=np.arange(1, num_labels + 1)
    )

    # Get the maximum value in the labels
    # print("Get the maximum value in the labels")
    # values = ndimage.measurements.maximum(
    #     img, labels=labels, index=np.arange(1, num_labels + 1)
    # )
    return coords

#Plotting
def plot_eigen_psfs_z_by_pc(eigen_psfs,i_len,j_len,psf_window):
    j_steps = np.floor_divide(psf_window[0], j_len)
    fig, ax = plt.subplots(i_len, j_len)
    for i in range(i_len):
        for j in range(j_len):
            ax[i, j].imshow(eigen_psfs[i, j * j_steps, :, :])
            # ax[i,0].imshow(eigen_psfs[1,0,:,:])
            # ax[0,1].imshow(eigen_psfs[0,10,:,:])
            # ax[1,1].imshow(eigen_psfs[1,10,:,:])
    return fig

def plot_eigen_psfs_z_proj_by_pc(eigen_psfs,i_len):
    fig, ax = plt.subplots(1, i_len)
    for i in range(i_len):
        ax[i].imshow(np.max(eigen_psfs[i, :, :, :], axis=1))
        # ax[i,0].imshow(eigen_psfs[1,0,:,:])
        # ax[0,1].imshow(eigen_psfs[0,10,:,:])
        # ax[1,1].imshow(eigen_psfs[1,10,:,:])
    # plt.show()
    # plt.imshow(np.sum(eigen_psfs[0, :, :, :], axis=1))  # %%
    return fig

def __main__(psf_image, psf_centres, normalise=True, centre_crop=True):
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
