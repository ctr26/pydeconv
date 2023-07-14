# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import importlib
import pydeconv

from pydeconv import utils, point_spread_function
from pydeconv import deconvolve
from pydeconv.utils import xyz_viewer
import os
import pims
import numpy as np
import PIL
import pims
import matplotlib.pyplot as plt
from os.path import expanduser
import scipy.ndimage as ndimage
from skimage.exposure import equalize_adapthist

# from pydeconv import utils, point_spread_function


# %%
image_file = "~/+projects/2019_jrl/2019_jRL_impute/data/CraigDeconvolutionData/200904_14.02.08_Step_Size_-0.4_Wavelength_DAPI 452-45_FL-15_CUBIC-1_pipetteTipMount_8bit.tif"
image_file = "~/+projects/2019_jrl/2019_jRL_impute/data/CraigDeconvolutionData/8bit.tif"
image_file = (
    "~/+projects/2019_jrl/2019_jRL_impute/data/CraigDeconvolutionData/8bit_bf.tif"
)

beads_file = "~/+projects/2019_jrl/2019_jRL_impute/data/CraigDeconvolutionData/2020-09-04 - calibration/beads/200904_16.50.57_Step_Size_-0.4_Wavelength_DAPI 452-45_500nm_TetBeads/MMStack.ome.tif"
print("Expanding dir")
image_file = os.path.expanduser(image_file)
beads_file = os.path.expanduser(beads_file)


psf_window = (140, 40, 40)  # (z,y,x)
psf_window = (60, 20, 20)
example_psf_idx = 10

# psf_window = (40, 40, 140)  # (x,y,z)
# psf_window = (20, 20, 60)

# %% Get images
psf = pims.open(beads_file)
image = pims.open(image_file)
image_np = np.array(image)
# %%
psf_image_scaled = utils.scaleImage(psf, np.uint8)
# %% Getting coords
coords = point_spread_function.getPSFcoords(psf_image_scaled, psf_window)
# Crop PSFs
# %%
cropped, coord_list = utils.cropNDv(psf_image_scaled, centres=coords, window=psf_window)
# %%
utils.xyz_viewer(cropped[50, :, :, :])
plt.show()
# %%


point_spread_function.psf_histogram(cropped)
plt.show()
# %%
# def get_df_from_var_psf(cropped):
import pandas as pd

index = pd.MultiIndex.from_arrays(
    np.array(coord_list).transpose(), names=("z", "y", "x")
)
psf_df = pd.DataFrame(cropped.reshape(cropped.shape[0], -1)).dropna(0).set_index(index)

# return flat

# %%
importlib.reload(pydeconv.point_spread_function)

psf_df = point_spread_function.get_df_from_var_psf(cropped, coord_list)
psf_df.sum(axis=1).hist()
psf_df_cleaned = point_spread_function.remove_outliers_sigma(psf_df)
df = psf_df_cleaned
df_normalised = point_spread_function.normalise_df(df)

ssim = point_spread_function.get_ssim_df(df, psf_window)
mse = point_spread_function.get_mse_df(df)
# df[] = point_spread_function.get_mse_df(df)
df = df_normalised
# %%
df = df_normalised
importlib.reload(pydeconv.point_spread_function)

# outliers = point_spread_function.get_outliers_from_forest(df)
inliers = point_spread_function.get_inliers_from_forest(df)
df = df[inliers]
# %%
example_psf = point_spread_function.get_image_from_df(df, example_psf_idx, psf_window)
plt.imshow(np.max(example_psf, axis=1))
# %%
SINGLE_PEAK_FILTERING = False
if SINGLE_PEAK_FILTERING:
    inliers = point_spread_function.inliers_by_single_peak(df, psf_window)
    df = df[inliers]
    example_psf = point_spread_function.get_image_from_df(df, 0, psf_window)
    plt.imshow(np.max(example_psf, axis=1))
# %%
pca_df, pca = point_spread_function.psf_df_pca(df)
plt.scatter(pca_df[0], pca_df[1])
plt.show()
# pca.fit_transform(cropped)
n_components = 10
pc_component = 0

# include average
eigen_psfs = point_spread_function.eigen_psfs_from_pca(pca, psf_window)
# cum_sum_exp_var = np.cumsum(pca.explained_variance_ratio_)
accuracy = point_spread_function.accuracy_from_pca(pca)
# %%
importlib.reload(pydeconv.point_spread_function)

point_spread_function.plot_eigen_psfs_z_by_pc(eigen_psfs, 5, 5, psf_window)
plt.show()
point_spread_function.plot_eigen_psfs_z_proj_by_pc(eigen_psfs, 5)
plt.show()

# %%

plt.show()
plt.imshow(np.max(np.array(df.iloc[10, :]).reshape(psf_window), axis=1))
# %% Plot cumaltive explained variance

plt.plot(accuracy[0:n_components])
plt.title(f"{str(accuracy[n_components])}")
plt.show()

# %%
mu = df.mean(axis=0)
mu_tru = np.mean(df, axis=0).to_numpy().reshape(psf_window)
plt.imshow(np.max(mu_tru, axis=1))
# %%
# plt.imshow(np.max(mu.to_numpy().reshape(psf_window), axis=1))
# %%
flat_eigen = eigen_psfs.reshape(-1, np.prod(psf_window))
# %%
psf_at_idx = df.iloc[example_psf_idx, :].to_numpy().reshape(psf_window)
plt.imshow(np.max(psf_at_idx, axis=1))
plt.show()
# %%
point_spread_function.plot_reconstruction_scores(
    pca_df, mu, pca, psf_window, example_psf_idx, psf_at_idx, components=50
)
plt.show()
# nComp = 1

# %%

point_spread_function.plot_pca_weights(pca_df)
plt.show()

# %%
importlib.reload(pydeconv.point_spread_function)

z_coords, y_coords, x_coords, = point_spread_function.get_xyz_coord(
    pca_df, psf_image_scaled.shape
)
f = point_spread_function.get_pc_weighting_fun_2D(pca_df,pc_component)
c_i = np.log(f(x_coords, y_coords))

plt.imshow(c_i)
plt.show()

# %%

# x_list = pca_df.index.get_level_values("x")
# y_list = pca_df.index.get_level_values("y")
# z_list = pca_df.index.get_level_values("z")

rbfi = point_spread_function.get_pc_weighting_fun_3D(pca_df, pc_component)

# %%
importlib.reload(pydeconv.point_spread_function)
cropping_window = [62, 150, 150]
dask_xyz = point_spread_function.get_dask_grid_tuple_from_pca_df_3D(
    pca_df, cropping_window
)
# %%
from dask.diagnostics import ProgressBar


# with ProgressBar():
g = point_spread_function.map_dask_grid_to_rbf(dask_xyz, rbfi)

plt.show()
plt.imshow(g[0, :, :])

# %%

cropped_size = np.floor(np.array(image_np.shape) / 4).astype(int)
image_centre = np.divide(image_np.shape, 2)
image_min = image_centre - np.divide(np.ones_like(image_centre) * cropped_size, 2)
image_max = image_centre + np.divide(np.ones_like(image_centre) * cropped_size, 2)
# %%
# image_min, image_max = cropping

# x_range_psf = np.floor(np.arange(image_min[2], image_max[2])).astype(int)
# y_range_psf = np.floor(np.arange(image_min[1], image_max[1])).astype(int)
# z_range_psf = np.floor(np.arange(image_min[0], image_max[0])).astype(int)


# grid_psf = np.mgrid[
#     min(z_range_psf) : max(z_range_psf),
#     min(y_range_psf) : max(y_range_psf),
#     min(x_range_psf) : max(x_range_psf)
# ]


# grid_psf_new = np.mgrid[np.floor(image_min) : np.floor(image_max)]
# %%
importlib.reload(pydeconv.point_spread_function)

psf_cropping = [psf_window, cropped_size - psf_window]
cropping = [image_min, image_max]

grid_psf_cropped_grid = point_spread_function.get_cropping_grid(psf_cropping)
small_image_grid = point_spread_function.get_cropping_grid(cropping)

pc_component_weight_map = point_spread_function.get_pc_weighting_map_3D(
    pca_df,
    small_image_grid,
    pc_component,
)

utils.xyz_viewer(pc_component_weight_map)
# %%


pc_component = 0
melt_df_slim = pd.melt(
    pca_df[[pc_component]],
    var_name="PC",
    value_name="Weight",
    ignore_index=False,
).reset_index()



from scipy.interpolate import Rbf
import dask.array as da
from dask.diagnostics import ProgressBar

# melt_df

# pc_idx,x_list,y_list,z_list = X["PC"],X["x"],X["y"],X["z"]

x_range = np.floor(np.arange(image_min[2], image_max[2])).astype(int)
y_range = np.floor(np.arange(image_min[1], image_max[1])).astype(int)
z_range = np.floor(np.arange(image_min[0], image_max[0])).astype(int)

grid_mesh = np.mgrid[
    min(z_range) : max(z_range),
    min(y_range) : max(y_range),
    min(x_range) : max(x_range)
]

image_min_psf = psf_window
image_max_psf = cropped_size - psf_window

x_range_psf = np.floor(np.arange(image_min_psf[2], image_max_psf[2])).astype(int)
y_range_psf = np.floor(np.arange(image_min_psf[1], image_max_psf[1])).astype(int)
z_range_psf = np.floor(np.arange(image_min_psf[0], image_max_psf[0])).astype(int)
# grid_z, grid_y, grid_x = np.meshgrid(x_range,y_range, z_range)


dask_xyzp = da.from_array(grid_mesh, chunks=5)

rbfi = Rbf(
    melt_df_slim["z"],
    melt_df_slim["y"],
    melt_df_slim["x"],
    melt_df_slim["Weight"],
    function="gaussian",
    smooth = 0
)


with ProgressBar():
    f = da.map_blocks(
        rbfi,
        *dask_xyzp
    )
    # g = client.persist(f)
    pc_component_weight_map = f.compute()
# %%
small_image = image_np[small_image_grid[0], small_image_grid[1], small_image_grid[2]]
image_crop = small_image

# small_image = image_np[small_image_grid]

# utils.xyz_viewer(small_image)

# cropped_image_crop = np.zeros_like(image_np)[grid_psf[0],grid_psf[1],grid_psf[2]]

fig = utils.xyz_viewer(image_crop)
# %%
importlib.reload(pydeconv.deconvolve)
zero_pc = np.zeros_like(pc_component_weight_map)
deconvolved_image = deconvolve.richardson_lucy_varying_1PC(
    small_image,
    grid_psf_cropped_grid,
    eigen_psfs[0, :, :, :],
    zero_pc,
    mu_tru,
    iterations=25,
)
# %%

from scipy.signal import convolve
cropped_image = small_image
grid_psf = grid_psf_cropped_grid

y = cropped_image
x_t = np.full(y.shape, 0.5)
x_t = cropped_image.copy()

c_0 = np.array(mu).reshape(*psf_window)
c_1 = eigen_psfs[pc_component, :, :, :] 
c_1 = np.ones_like(c_0)

c_0_T = np.flip(c_0)
c_1_T = np.flip(c_1)

d_0 = np.ones_like(y)
d_1 = pc_component_weight_map
d_1 = np.zeros_like(d_0)

d_0_T = d_0
d_1_T = d_1
x_t_crop = x_t[grid_psf[0],grid_psf[1],grid_psf[2]]

utils.xyz_viewer(x_t_crop)
for i in range(25):
    c_0_x_t = convolve(x_t, c_0,  mode="same")
    c_1_x_t = convolve(x_t, c_1,  mode="same")

    d_0_c_0_x_t =  np.multiply(d_0,c_0_x_t)
    d_1_c_1_x_t =  np.multiply(d_1,c_1_x_t)

    d_c_x_t = np.add(d_0_c_0_x_t,d_1_c_1_x_t)
    y_over_d_c_x_t = np.divide(y,d_c_x_t)

    c_0_T_y_over_d_c_x_t = convolve(y_over_d_c_x_t,c_0_T, mode="same")
    c_1_T_y_over_d_c_x_t = convolve(y_over_d_c_x_t,c_1_T, mode="same")

    d_0_T_c_0_y_over_d_c_x_t = np.multiply(d_0_T,c_0_T_y_over_d_c_x_t)
    d_1_T_c_1_y_over_d_c_x_t = np.multiply(d_1_T,c_1_T_y_over_d_c_x_t)

    d_c_y_over_d_c_x_t = np.add(d_0_T_c_0_y_over_d_c_x_t,d_1_T_c_1_y_over_d_c_x_t)

    ones = np.ones_like(y)

    c_0_T_ones = convolve(ones,c_0_T, mode="same")
    c_1_T_ones = convolve(ones,c_1_T, mode="same")

    d_0_T_c_0_T_ones = np.multiply(d_0_T,c_0_T_ones)
    d_1_T_c_1_T_ones = np.multiply(d_1_T,c_1_T_ones)

    d_T_c_T_ones = np.add(d_0_T_c_0_T_ones,d_1_T_c_1_T_ones)
    d_c_y_over_d_c_x_t_over_d_T_c_T_ones = np.divide(d_c_y_over_d_c_x_t,d_T_c_T_ones)
    x_t = np.multiply(x_t,d_c_y_over_d_c_x_t_over_d_T_c_T_ones)
    # xyz_viewer(x_t)
    x_t_crop = x_t[grid_psf[0],grid_psf[1],grid_psf[2]]
    utils.xyz_viewer(x_t_crop)
    plt.show()
    print(f"Iteration {str(i)}")
# %%

from scipy.signal import convolve
cropped_image = small_image
grid_psf = grid_psf_cropped_grid

y = cropped_image
x_t = np.full(y.shape, 0.5)
x_t = cropped_image.copy()

c_0 = np.array(mu).reshape(*psf_window)

c_0_T = np.flip(c_0)

# d_0 = np.ones_like(y)

# d_0_T = d_0
x_t_crop = x_t[grid_psf[0],grid_psf[1],grid_psf[2]]

utils.xyz_viewer(x_t_crop)
for i in range(25):
    c_0_x_t = convolve(x_t, c_0,  mode="same")
    # c_1_x_t = convolve(x_t, c_1,  mode="same")

    d_0_c_0_x_t = c_0_x_t
    # d_1_c_1_x_t =  np.multiply(d_1,c_1_x_t)

    d_c_x_t = d_0_c_0_x_t
    y_over_d_c_x_t = np.divide(y,d_c_x_t)

    c_0_T_y_over_d_c_x_t = convolve(y_over_d_c_x_t,c_0_T, mode="same")
    # c_1_T_y_over_d_c_x_t = convolve(y_over_d_c_x_t,c_1_T, mode="same")

    d_0_T_c_0_y_over_d_c_x_t = c_0_T_y_over_d_c_x_t
    # d_1_T_c_1_y_over_d_c_x_t = np.multiply(d_1_T,c_1_T_y_over_d_c_x_t)

    d_c_y_over_d_c_x_t = d_0_T_c_0_y_over_d_c_x_t

    ones = np.ones_like(y)

    c_0_T_ones = convolve(ones,c_0_T, mode="same")
    # c_1_T_ones = convolve(ones,c_1_T, mode="same")

    d_0_T_c_0_T_ones = np.multiply(d_0_T,c_0_T_ones)
    # d_1_T_c_1_T_ones = np.multiply(d_1_T,c_1_T_ones)

    d_T_c_T_ones = d_0_T_c_0_T_ones
    d_c_y_over_d_c_x_t_over_d_T_c_T_ones = np.divide(d_c_y_over_d_c_x_t,d_T_c_T_ones)
    x_t = np.multiply(x_t,d_c_y_over_d_c_x_t_over_d_T_c_T_ones)
    # xyz_viewer(x_t)
    x_t_crop = x_t[grid_psf[0],grid_psf[1],grid_psf[2]]
    utils.xyz_viewer(x_t_crop)
    plt.show()
    print(f"Iteration {str(i)}")
# %%
# def cropNDv(img, centres, window=[20, 40, 40]):
# img = psf_image_scaled
# window = psf_window
# centres = coords
# %%
# cropped_list = np.full((len(centres), *window), np.nan)
# i = 0
# centres_list = []
# centre = centres[100]

# %%


melt_df = pd.melt(
    pca_df.iloc[:, 0:n_components],
    var_name="PC",
    value_name="Weight",
    ignore_index=False,
).reset_index()

from scipy.interpolate import Rbf
import dask.array as da
from dask.diagnostics import ProgressBar

# melt_df
x_list = pca_df.index.get_level_values("x")
y_list = pca_df.index.get_level_values("y")
z_list = pca_df.index.get_level_values("z")
# pc_idx,x_list,y_list,z_list = X["PC"],X["x"],X["y"],X["z"]
rbfi = Rbf(
    melt_df["PC"],
    melt_df["z"],
    melt_df["y"],
    melt_df["x"],
    melt_df["Weight"],
    function="cubic",
)

grid_pc, grid_z, grid_y, grid_x = np.mgrid[
    0:n_components,
    min(z_list) : max(z_list) : 50j,
    min(y_list) : max(y_list) : 100j,
    min(x_list) : max(x_list) : 100j,
]

grid_pc, grid_z, grid_y, grid_x = np.mgrid[
    0:n_components,
    min(z_list) : max(z_list) : 3j,
    min(y_list) : max(y_list) : 100j,
    min(x_list) : max(x_list) : 100j,
]

dask_xyzp = da.from_array((grid_pc, grid_z, grid_y, grid_x), chunks=5)
# %%
# Slow
# from dask.distributed import progress

# with ProgressBar():
#     f = da.map_blocks(
#         rbfi,
#         dask_xyzp[0, :, :, :],
#         dask_xyzp[1, :, :, :],
#         dask_xyzp[2, :, :, :],
#         dask_xyzp[3, :, :, :],
#     )
#     # g = client.persist(f)
#     g = f.compute()
#     # g = progress(client.persist(f))


# %%
# plt.imshow(g[0, 25, :, :])
plt.show()
plt.imshow(g[0, 0, :, :])
# %% Deconvolution

# make richardson lucy function that takes in radial basis function

# mu = np.mean(df, axis=1).reshape(window)
mu = np.mean(df, axis=0).to_numpy().reshape(psf_window)
plt.imshow(np.max(mu, axis=1))
# %%
# eigen_psfs.flatten()
# flat_eigen = eigen_psfs.reshape(-1,np.prod(psf_window))

# psf_image_first = df.iloc[10,:].to_numpy().reshape(psf_window)
# plt.imshow(np.max(psf_image_first, axis=1))
# plt.show()
# flat_eigen_psfs = flat_eigen
# pc_weightings =pca_df.iloc[10,:]

# reconstructed_psf = np.dot(pc_weightings,flat_eigen_psfs).reshape(psf_window)
# plt.imshow(np.max(reconstructed_psf, axis=1))

# Xhat = np.dot(pca_df, eigen_psfs)
# Xhat += mu

# %%
# z, y, x = coord_list[0], coord_list[1], coord_list[2]
cropped_size = [62, 150, 150]
cropped_size = np.floor(np.array(image_np.shape) / 4).astype(int)
image_centre = np.divide(image_np.shape, 2)
image_min = image_centre - np.divide(np.ones_like(image_centre) * cropped_size, 2)
image_max = image_centre + np.divide(np.ones_like(image_centre) * cropped_size, 2)
# %%
# x_range = np.arange(np.floor(np.divide(image_np.shape[2], 20)).astype(int))
# y_range = np.arange(np.floor(np.divide(image_np.shape[1], 20)).astype(int))
# z_range = np.arange(np.floor(np.divide(image_np.shape[0], 20)).astype(int))


x_range = np.floor(np.arange(image_min[2], image_max[2])).astype(int)
y_range = np.floor(np.arange(image_min[1], image_max[1])).astype(int)
z_range = np.floor(np.arange(image_min[0], image_max[0])).astype(int)

psf_window_half = np.divide(psf_window, 2)

image_min_psf = psf_window
image_max_psf = cropped_size - psf_window
# image_min_psf = image_min+psf_window_half
# image_max_psf = image_max-psf_window_half


x_range_psf = np.floor(np.arange(image_min_psf[2], image_max_psf[2])).astype(int)
y_range_psf = np.floor(np.arange(image_min_psf[1], image_max_psf[1])).astype(int)
z_range_psf = np.floor(np.arange(image_min_psf[0], image_max_psf[0])).astype(int)
# grid_z, grid_y, grid_x = np.meshgrid(x_range,y_range, z_range)

# grid_pc, grid_z, grid_y, grid_x = np.mgrid[
#     0:1,
#     min(z_range) : max(z_range),
#     min(y_range) : max(y_range),
#     min(x_range) : max(x_range),
# ]


grid_z_psf, grid_y_psf, grid_x_psf = np.mgrid[
    min(z_range_psf) : max(z_range_psf),
    min(y_range_psf) : max(y_range_psf),
    min(x_range_psf) : max(x_range_psf),
]

grid_psf = np.mgrid[
    min(z_range_psf) : max(z_range_psf),
    min(y_range_psf) : max(y_range_psf),
    min(x_range_psf) : max(x_range_psf),
]


print(grid_x_psf.shape)
# [0],

#
# # pcs =
# principle_components = rbfi[:,z,y,x]
# reconstructed_psf_1 = principle_components[0]*eigen_psfs[0]

# z_grid, y_grid, x_grid = np.meshgrid( np.divide(image_np.shape,4)

# z_grid, y_grid, x_grid = np.meshgrid(z_range, y_range, z_range)
# pcs = np.zeros_like(x_grid)
# reconstructed_psf = principle_components
# principle_components = rbfi(pcs, z_grid, y_grid, x_grid)


dask_xyzp = da.from_array((grid_pc, grid_z, grid_y, grid_x), chunks=5)
# %%
# Slow full

pc_component = 0
melt_df_slim = pd.melt(
    pca_df[[pc_component]],
    var_name="PC",
    value_name="Weight",
    ignore_index=False,
).reset_index()



from scipy.interpolate import Rbf
import dask.array as da
from dask.diagnostics import ProgressBar

# melt_df

# pc_idx,x_list,y_list,z_list = X["PC"],X["x"],X["y"],X["z"]

grid_mesh = np.mgrid[
    min(z_range) : max(z_range),
    min(y_range) : max(y_range),
    min(x_range) : max(x_range)
]

x_range_psf = np.floor(np.arange(image_min_psf[2], image_max_psf[2])).astype(int)
y_range_psf = np.floor(np.arange(image_min_psf[1], image_max_psf[1])).astype(int)
z_range_psf = np.floor(np.arange(image_min_psf[0], image_max_psf[0])).astype(int)
# grid_z, grid_y, grid_x = np.meshgrid(x_range,y_range, z_range)


dask_xyzp = da.from_array(grid_mesh, chunks=5)

rbfi = Rbf(
    melt_df_slim["z"],
    melt_df_slim["y"],
    melt_df_slim["x"],
    melt_df_slim["Weight"],
    function="cubic",
)

with ProgressBar():
    f = da.map_blocks(
        rbfi,
        *dask_xyzp
    )
    # g = client.persist(f)
    pc_component_weight_map = f.compute()

xyz_viewer(pc_component_weight_map)

cropped_image = image_np[grid_mesh[0],grid_mesh[1],grid_mesh[2]]
xyz_viewer(cropped_image)

cropped_image_crop = cropped_image[grid_psf[0],grid_psf[1],grid_psf[2]]
# cropped_image_crop = np.zeros_like(image_np)[grid_psf[0],grid_psf[1],grid_psf[2]]

xyz_viewer(cropped_image_crop)
# %%

melt_df = pd.melt(
    pca_df.iloc[:, 0:n_components],
    var_name="PC",
    value_name="Weight",
    ignore_index=False,
).reset_index()

from scipy.interpolate import Rbf
import dask.array as da
from dask.diagnostics import ProgressBar

# melt_df

# pc_idx,x_list,y_list,z_list = X["PC"],X["x"],X["y"],X["z"]
rbfi = Rbf(
    melt_df["PC"],
    melt_df["z"],
    melt_df["y"],
    melt_df["x"],
    melt_df["Weight"],
    function="cubic",
)

grid_pc, grid_z, grid_y, grid_x = np.mgrid[
    0:n_components,
    min(z_list) : max(z_list) : 50j,
    min(y_list) : max(y_list) : 100j,
    min(x_list) : max(x_list) : 100j,
]

grid_pc, grid_z, grid_y, grid_x = np.mgrid[
    0:n_components,
    min(z_list) : max(z_list) : 3j,
    min(y_list) : max(y_list) : 100j,
    min(x_list) : max(x_list) : 100j,
]

dask_xyzp = da.from_array((grid_pc, grid_z, grid_y, grid_x), chunks=5)
# %%
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
    # g = progress(client.persist(f))


# %%
# plt.imshow(g[0, 25, :, :])
plt.show()
plt.imshow(g[0, 0, :, :])
# %% Deconvolution

# make richardson lucy function that takes in radial basis function

# mu = np.mean(df, axis=1).reshape(window)
mu = np.mean(df, axis=0).to_numpy().reshape(psf_window)
plt.imshow(np.max(mu, axis=1))
# %%
# eigen_psfs.flatten()
# flat_eigen = eigen_psfs.reshape(-1,np.prod(psf_window))

# psf_image_first = df.iloc[10,:].to_numpy().reshape(psf_window)
# plt.imshow(np.max(psf_image_first, axis=1))
# plt.show()
# flat_eigen_psfs = flat_eigen
# pc_weightings =pca_df.iloc[10,:]

# reconstructed_psf = np.dot(pc_weightings,flat_eigen_psfs).reshape(psf_window)
# plt.imshow(np.max(reconstructed_psf, axis=1))

# Xhat = np.dot(pca_df, eigen_psfs)
# Xhat += mu

# %%
# z, y, x = coord_list[0], coord_list[1], coord_list[2]
cropped_size = [62, 150, 150]
cropped_size = np.floor(np.array(image_np.shape) / 4).astype(int)
image_centre = np.divide(image_np.shape, 2)
image_min = image_centre - np.divide(np.ones_like(image_centre) * cropped_size, 2)
image_max = image_centre + np.divide(np.ones_like(image_centre) * cropped_size, 2)
# %%
# x_range = np.arange(np.floor(np.divide(image_np.shape[2], 20)).astype(int))
# y_range = np.arange(np.floor(np.divide(image_np.shape[1], 20)).astype(int))
# z_range = np.arange(np.floor(np.divide(image_np.shape[0], 20)).astype(int))


x_range = np.floor(np.arange(image_min[2], image_max[2])).astype(int)
y_range = np.floor(np.arange(image_min[1], image_max[1])).astype(int)
z_range = np.floor(np.arange(image_min[0], image_max[0])).astype(int)

psf_window_half = np.divide(psf_window, 2)

image_min_psf = psf_window
image_max_psf = cropped_size - psf_window
# image_min_psf = image_min+psf_window_half
# image_max_psf = image_max-psf_window_half


x_range_psf = np.floor(np.arange(image_min_psf[2], image_max_psf[2])).astype(int)
y_range_psf = np.floor(np.arange(image_min_psf[1], image_max_psf[1])).astype(int)
z_range_psf = np.floor(np.arange(image_min_psf[0], image_max_psf[0])).astype(int)
# grid_z, grid_y, grid_x = np.meshgrid(x_range,y_range, z_range)

# grid_pc, grid_z, grid_y, grid_x = np.mgrid[
#     0:1,
#     min(z_range) : max(z_range),
#     min(y_range) : max(y_range),
#     min(x_range) : max(x_range),
# ]


grid_z_psf, grid_y_psf, grid_x_psf = np.mgrid[
    min(z_range_psf) : max(z_range_psf),
    min(y_range_psf) : max(y_range_psf),
    min(x_range_psf) : max(x_range_psf),
]

grid_psf = np.mgrid[
    min(z_range_psf) : max(z_range_psf),
    min(y_range_psf) : max(y_range_psf),
    min(x_range_psf) : max(x_range_psf),
]


print(grid_x_psf.shape)
# [0],

#
# # pcs =
# principle_components = rbfi[:,z,y,x]
# reconstructed_psf_1 = principle_components[0]*eigen_psfs[0]

# z_grid, y_grid, x_grid = np.meshgrid( np.divide(image_np.shape,4)

# z_grid, y_grid, x_grid = np.meshgrid(z_range, y_range, z_range)
# pcs = np.zeros_like(x_grid)
# reconstructed_psf = principle_components
# principle_components = rbfi(pcs, z_grid, y_grid, x_grid)


dask_xyzp = da.from_array((grid_pc, grid_z, grid_y, grid_x), chunks=5)
# %%
# Slow full

pc_component = 0
melt_df_slim = pd.melt(
    pca_df[[pc_component]],
    var_name="PC",
    value_name="Weight",
    ignore_index=False,
).reset_index()



from scipy.interpolate import Rbf
import dask.array as da
from dask.diagnostics import ProgressBar

# melt_df

# pc_idx,x_list,y_list,z_list = X["PC"],X["x"],X["y"],X["z"]

grid_mesh = np.mgrid[
    min(z_range) : max(z_range),
    min(y_range) : max(y_range),
    min(x_range) : max(x_range)
]

x_range_psf = np.floor(np.arange(image_min_psf[2], image_max_psf[2])).astype(int)
y_range_psf = np.floor(np.arange(image_min_psf[1], image_max_psf[1])).astype(int)
z_range_psf = np.floor(np.arange(image_min_psf[0], image_max_psf[0])).astype(int)
# grid_z, grid_y, grid_x = np.meshgrid(x_range,y_range, z_range)


dask_xyzp = da.from_array(grid_mesh, chunks=5)

rbfi = Rbf(
    melt_df_slim["z"],
    melt_df_slim["y"],
    melt_df_slim["x"],
    melt_df_slim["Weight"],
    function="cubic",
)

with ProgressBar():
    f = da.map_blocks(
        rbfi,
        *dask_xyzp
    )
    # g = client.persist(f)
    pc_component_weight_map = f.compute()

xyz_viewer(pc_component_weight_map)

cropped_image = image_np[grid_mesh[0],grid_mesh[1],grid_mesh[2]]
xyz_viewer(cropped_image)

cropped_image_crop = cropped_image[grid_psf[0],grid_psf[1],grid_psf[2]]
# cropped_image_crop = np.zeros_like(image_np)[grid_psf[0],grid_psf[1],grid_psf[2]]

xyz_viewer(cropped_image_crop)

# %% 0+1 Components RL
# Expanded the modified richardson lucy equation to the first two components.
from scipy.signal import convolve

y = cropped_image
x_t = np.full(y.shape, 0.5)
x_t = cropped_image.copy()

c_0 = np.array(mu).reshape(*psf_window)
c_1 = eigen_psfs[pc_component, :, :, :] 

c_0_T = np.flip(c_0)
c_1_T = np.flip(c_1)

d_0 = np.ones_like(y)
d_1 = pc_component_weight_map
d_1 = np.zeros_like(d_0)

d_0_T = d_0
d_1_T = d_1
x_t_crop = x_t[grid_psf[0],grid_psf[1],grid_psf[2]]
xyz_viewer(x_t_crop)
for i in range(25):
    c_0_x_t = convolve(x_t, c_0,  mode="same")
    c_1_x_t = convolve(x_t, c_1,  mode="same")

    d_0_c_0_x_t =  np.multiply(d_0,c_0_x_t)
    d_1_c_1_x_t =  np.multiply(d_1,c_1_x_t)

    d_c_x_t = np.add(d_0_c_0_x_t,d_1_c_1_x_t)
    y_over_d_c_x_t = np.divide(y,d_c_x_t)

    c_0_T_y_over_d_c_x_t = convolve(y_over_d_c_x_t,c_0_T, mode="same")
    c_1_T_y_over_d_c_x_t = convolve(y_over_d_c_x_t,c_1_T, mode="same")

    d_0_T_c_0_y_over_d_c_x_t = np.multiply(d_0_T,c_0_T_y_over_d_c_x_t)
    d_1_T_c_1_y_over_d_c_x_t = np.multiply(d_1_T,c_1_T_y_over_d_c_x_t)

    d_c_y_over_d_c_x_t = np.add(d_0_T_c_0_y_over_d_c_x_t,d_1_T_c_1_y_over_d_c_x_t)

    ones = np.ones_like(y)

    c_0_T_ones = convolve(ones,c_0_T, mode="same")
    c_1_T_ones = convolve(ones,c_1_T, mode="same")

    d_0_T_c_0_T_ones = np.multiply(d_0_T,c_0_T_ones)
    d_1_T_c_1_T_ones = np.multiply(d_1_T,c_1_T_ones)

    d_T_c_T_ones = np.add(d_0_T_c_0_T_ones,d_1_T_c_1_T_ones)
    d_c_y_over_d_c_x_t_over_d_T_c_T_ones = np.divide(d_c_y_over_d_c_x_t,d_T_c_T_ones)
    x_t = np.multiply(x_t,d_c_y_over_d_c_x_t_over_d_T_c_T_ones)
    # xyz_viewer(x_t)
    x_t_crop = x_t[grid_psf[0],grid_psf[1],grid_psf[2]]
    xyz_viewer(x_t_crop)
    plt.show()
    print(f"Iteration {str(i)}")


# centre = np.array(centre)
# centre_dist = np.divide(window, 2)
# shape = img.shape
# crops = []
# # for i, dim in enumerate(shape):
# #     l = (centre[-(i + 1)] - np.floor(centre_dist[-(i + 1)])).astype(int)
# #     r = (centre[-(i + 1)] + np.ceil(centre_dist[-(i + 1)])).astype(int)

# x_l = (centre[0] - np.floor(centre_dist[0])).astype(int)
# x_r = (centre[0] + np.ceil(centre_dist[0])).astype(int)


# for centre in centres:
#     try:
#         cropped_list[i, :, :, :] = utils.cropND(img, centre, window=window)
#         centres_list.append(centres[i])
#         i += 1
#     except:
#         None
# cropped_list, centres_list

# %%

# %% Show crops


# %% Show bead_histogram

# %%
# eigen_psfs.flatten()
# flat_eigen = eigen_psfs.reshape(-1,np.prod(psf_window))

# psf_image_first = df.iloc[10,:].to_numpy().reshape(psf_window)
# plt.imshow(np.max(psf_image_first, axis=1))
# plt.show()
# flat_eigen_psfs = flat_eigen
# pc_weightings =pca_df.iloc[10,:]

# reconstructed_psf = np.dot(pc_weightings,flat_eigen_psfs).reshape(psf_window)
# plt.imshow(np.max(reconstructed_psf, axis=1))

# Xhat = np.dot(pca_df, eigen_psfs)
# Xhat += mu

# %%
# z, y, x = coord_list[0], coord_list[1], coord_list[2]
# cropped_size = [62, 150, 150]

# %%
# x_range = np.arange(np.floor(np.divide(image_np.shape[2], 20)).astype(int))
# y_range = np.arange(np.floor(np.divide(image_np.shape[1], 20)).astype(int))
# z_range = np.arange(np.floor(np.divide(image_np.shape[0], 20)).astype(int))

# x_range = np.floor(np.arange(image_min[2], image_max[2])).astype(int)
# y_range = np.floor(np.arange(image_min[1], image_max[1])).astype(int)
# z_range = np.floor(np.arange(image_min[0], image_max[0])).astype(int)

# psf_window_half = np.divide(psf_window, 2)

# image_min_psf = psf_window
# image_max_psf = cropped_size - psf_window

# # image_min_psf = image_min+psf_window_half
# # image_max_psf = image_max-psf_window_half


# x_range_psf = np.floor(np.arange(image_min_psf[2], image_max_psf[2])).astype(int)
# y_range_psf = np.floor(np.arange(image_min_psf[1], image_max_psf[1])).astype(int)
# z_range_psf = np.floor(np.arange(image_min_psf[0], image_max_psf[0])).astype(int)
# # grid_z, grid_y, grid_x = np.meshgrid(x_range,y_range, z_range)

# # grid_pc, grid_z, grid_y, grid_x = np.mgrid[
# #     0:1,
# #     min(z_range) : max(z_range),
# #     min(y_range) : max(y_range),
# #     min(x_range) : max(x_range),
# # ]


# grid_z_psf, grid_y_psf, grid_x_psf = np.mgrid[
#     min(z_range_psf) : max(z_range_psf),
#     min(y_range_psf) : max(y_range_psf),
#     min(x_range_psf) : max(x_range_psf),
# ]

# grid_psf = np.mgrid[
#     min(z_range_psf) : max(z_range_psf),
#     min(y_range_psf) : max(y_range_psf),
#     min(x_range_psf) : max(x_range_psf),
# ]


# print(grid_x_psf.shape)
# # [0],

#
# # pcs =
# principle_components = rbfi[:,z,y,x]
# reconstructed_psf_1 = principle_components[0]*eigen_psfs[0]

# z_grid, y_grid, x_grid = np.meshgrid( np.divide(image_np.shape,4)

# z_grid, y_grid, x_grid = np.meshgrid(z_range, y_range, z_range)
# pcs = np.zeros_like(x_grid)
# reconstructed_psf = principle_components
# principle_components = rbfi(pcs, z_grid, y_grid, x_grid)


# dask_xyzp = da.from_array((grid_pc, grid_z, grid_y, grid_x), chunks=5)
# %%
# Slow full

# melt_df_slim = pd.melt(
#     pca_df[[pc_component]],
#     var_name="PC",
#     value_name="Weight",
#     ignore_index=False,
# ).reset_index()


# from scipy.interpolate import Rbf
# import dask.array as da
# from dask.diagnostics import ProgressBar

# # melt_df

# # pc_idx,x_list,y_list,z_list = X["PC"],X["x"],X["y"],X["z"]

# grid_mesh = np.mgrid[
#     min(z_range) : max(z_range),
#     min(y_range) : max(y_range),
#     min(x_range) : max(x_range)
# ]

# x_range_psf = np.floor(np.arange(image_min_psf[2], image_max_psf[2])).astype(int)
# y_range_psf = np.floor(np.arange(image_min_psf[1], image_max_psf[1])).astype(int)
# z_range_psf = np.floor(np.arange(image_min_psf[0], image_max_psf[0])).astype(int)
# # grid_z, grid_y, grid_x = np.meshgrid(x_range,y_range, z_range)


# dask_xyzp = da.from_array(grid_mesh, chunks=5)

# rbfi = Rbf(
#     melt_df_slim["z"],
#     melt_df_slim["y"],
#     melt_df_slim["x"],
#     melt_df_slim["Weight"],
#     function="cubic",
# )

# with ProgressBar():
#     f = da.map_blocks(rbfi, *dask_xyzp)
#     # g = client.persist(f)
#     pc_component_weight_map = f.compute()


# %% 0+1 Components RL
# Expanded the modified richardson lucy equation to the first two components.
# from scipy.signal import convolve

# y = cropped_image
# x_t = np.full(y.shape, 0.5)
# x_t = cropped_image.copy()

# c_0 = np.array(mu).reshape(*psf_window)
# c_1 = eigen_psfs[pc_component, :, :, :]

# c_0_T = np.flip(c_0)
# c_1_T = np.flip(c_1)

# d_0 = np.ones_like(y)
# d_1 = pc_component_weight_map

# d_0_T = d_0
# d_1_T = d_1
# x_t_crop = x_t[grid_psf[0], grid_psf[1], grid_psf[2]]
# xyz_viewer(x_t_crop)
# for i in range(25):
#     c_0_x_t = convolve(x_t, c_0, mode="same")
#     c_1_x_t = convolve(x_t, c_1, mode="same")

#     d_0_c_0_x_t = np.multiply(d_0, c_0_x_t)
#     d_1_c_1_x_t = np.multiply(d_1, c_1_x_t)

#     d_c_x_t = np.add(d_0_c_0_x_t, d_1_c_1_x_t)
#     y_over_d_c_x_t = np.divide(y, d_c_x_t)

#     c_0_T_y_over_d_c_x_t = convolve(y_over_d_c_x_t, c_0_T, mode="same")
#     c_1_T_y_over_d_c_x_t = convolve(y_over_d_c_x_t, c_1_T, mode="same")

#     d_0_T_c_0_y_over_d_c_x_t = np.multiply(d_0_T, c_0_T_y_over_d_c_x_t)
#     d_1_T_c_1_y_over_d_c_x_t = np.multiply(d_1_T, c_1_T_y_over_d_c_x_t)

#     d_c_y_over_d_c_x_t = np.add(d_0_T_c_0_y_over_d_c_x_t, d_1_T_c_1_y_over_d_c_x_t)

#     ones = np.ones_like(y)

#     c_0_T_ones = convolve(ones, c_0_T, mode="same")
#     c_1_T_ones = convolve(ones, c_1_T, mode="same")

#     d_0_T_c_0_T_ones = np.multiply(d_0_T, c_0_T_ones)
#     d_1_T_c_1_T_ones = np.multiply(d_1_T, c_1_T_ones)

#     d_T_c_T_ones = np.add(d_0_T_c_0_T_ones, d_1_T_c_1_T_ones)
#     d_c_y_over_d_c_x_t_over_d_T_c_T_ones = np.divide(d_c_y_over_d_c_x_t, d_T_c_T_ones)
#     x_t = np.multiply(x_t, d_c_y_over_d_c_x_t_over_d_T_c_T_ones)
#     # xyz_viewer(x_t)
#     x_t_crop = x_t[grid_psf[0], grid_psf[1], grid_psf[2]]
#     xyz_viewer(x_t_crop)
# # %%


# def richardson_lucy(image, psf, iterations=40, x_0=None):
#     return image_rl


# def richardson_lucy_varying(image, psf, iterations=40, x_0=None):
#     return image_rl


# for i in range(25):
#     c_0_x_t = convolve(x_t, c_0, mode="same")
#     c_1_x_t = convolve(x_t, c_1, mode="same")

#     d_0_c_0_x_t = np.multiply(d_0, c_0_x_t)
#     d_1_c_1_x_t = np.multiply(d_1, c_1_x_t)

#     # d_c_x_t = np.add.accumulate([d_0_c_0_x_t,d_1_c_1_x_t])
#     y_over_d_c_x_t = np.divide(y, d_c_x_t)

#     c_0_T_y_over_d_c_x_t = convolve(y_over_d_c_x_t, c_0_T, mode="same")
#     c_1_T_y_over_d_c_x_t = convolve(y_over_d_c_x_t, c_1_T, mode="same")

#     d_0_T_c_0_y_over_d_c_x_t = np.multiply(d_0_T, c_0_T_y_over_d_c_x_t)
#     d_1_T_c_1_y_over_d_c_x_t = np.multiply(d_1_T, c_1_T_y_over_d_c_x_t)

#     d_c_y_over_d_c_x_t = np.add(d_0_T_c_0_y_over_d_c_x_t, d_1_T_c_1_y_over_d_c_x_t)

#     ones = np.ones_like(y)

#     c_0_T_ones = convolve(ones, c_0_T, mode="same")
#     c_1_T_ones = convolve(ones, c_1_T, mode="same")

#     d_0_T_c_0_T_ones = np.multiply(d_0_T, c_0_T_ones)
#     d_1_T_c_1_T_ones = np.multiply(d_1_T, c_1_T_ones)

#     d_T_c_T_ones = np.add(d_0_T_c_0_T_ones, d_1_T_c_1_T_ones)

#     # d_c_x_t = np.add.accumulate([d_0_T_c_0_T_ones,d_1_T_c_1_T_ones])

#     d_c_y_over_d_c_x_t_over_d_T_c_T_ones = np.divide(d_c_y_over_d_c_x_t, d_T_c_T_ones)
#     x_t = np.multiply(x_t, d_c_y_over_d_c_x_t_over_d_T_c_T_ones)
#     # xyz_viewer(x_t)
#     x_t_crop = x_t[grid_psf[0], grid_psf[1], grid_psf[2]]
#     xyz_viewer(x_t_crop)
# #%%

# # np.save("data/pc_1_weighting.npy", g)
# # print("pc 1 weighting saved")

# # %%
# from scipy.signal import convolve

# pc_weighting_1 = pc_component_weight_map[0, :, :, :]
# pc_weighting_0 = np.ones_like(pc_weighting_1)

# image_rl_0 = cropped_image[0, :, :, :]
# image_rl_1 = cropped_image[0, :, :, :]


# psf_rl_0 = np.array(mu).reshape(*psf_window) / (np.array(mu).reshape(*psf_window).sum())
# psf_rl_1 = eigen_psfs[0, :, :, :] / psf_rl_0.sum()


# psf_rl_0 = np.array(mu).reshape(*psf_window)
# psf_rl_1 = eigen_psfs[0, :, :, :]


# # im_deconv =
# y = cropped_image[0, :, :, :]
# iterations = 40

# xyz_viewer(psf_rl_0)
# xyz_viewer(psf_rl_1)

# # plt.imshow(np.max(psf_rl_0, axis=0))
# # plt.show()
# # plt.imshow(np.max(psf_rl_0, axis=1))
# # plt.show()
# # plt.imshow(np.max(psf_rl_0, axis=2))


# # plt.imshow(np.max(psf_rl_1, axis=0))
# # plt.show()
# # plt.imshow(np.max(psf_rl_1, axis=1))
# # plt.show()
# # plt.imshow(np.max(psf_rl_1, axis=2))

# #%% One components
# prior_x = im_deconv = np.full(cropped_image[0, :, :, :].shape, 0.5)

# for i in range(4):
#     psf_mirror_0 = np.flip(psf_rl_0)
#     conv_0 = convolve(prior_x, psf_rl_0, mode="same")
#     weighted_conv_0 = np.multiply(pc_weighting_0, conv_0)
#     y_0 = np.divide(y, weighted_conv_0)
#     unweighted_belief_0 = np.multiply(pc_weighting_0, y_0)
#     conv_0_belief = convolve(unweighted_belief_0, psf_mirror_0, mode="same")
#     psf_normaliser_0 = convolve(np.ones_like(y), psf_mirror_0, mode="same")
#     psf_normaliser_0_weighted = np.multiply(psf_normaliser_0, pc_weighting_0)

#     unweighted_belief = unweighted_belief_0
#     psf_normaliser_weighted = psf_normaliser_0_weighted
#     # prior_x = (prior_x/psf_normaliser_weighted)*unweighted_belief
#     prior_x = (prior_x) * unweighted_belief
# # %% Correct
# prior_x = im_deconv = np.full(cropped_image[0, :, :, :].shape, 0.5)

# for i in range(40):
#     psf_mirror_0 = np.flip(psf_rl_0)
#     conv_0 = convolve(prior_x, psf_rl_0, mode="same")
#     # weighted_conv_0 = np.multiply(pc_weighting_0,conv_0)
#     y_0 = y / conv_0
#     # unweighted_belief_0 = np.multiply(pc_weighting_0,y_0)
#     conv_0_belief = convolve(y_0, psf_mirror_0, mode="same")
#     # psf_normaliser_0 = convolve(np.ones_like(y), psf_mirror_0, mode="same")
#     # psf_normaliser_0_weighted = np.multiply(psf_normaliser_0,pc_weighting_0)

#     # unweighted_belief = unweighted_belief_0
#     # psf_normaliser_weighted = psf_normaliser_0_weighted
#     # prior_x = (prior_x/psf_normaliser_weighted)*unweighted_belief
#     prior_x = (prior_x) * conv_0_belief
# # %%
# from scipy.signal import convolve, oaconvolve


# def rl(image, psf, iterations=50, clip=True, filter_epsilon=None):
#     float_type = np.promote_types(image.dtype, np.float32)
#     image = image.astype(float_type, copy=False)
#     psf = psf.astype(float_type, copy=False)
#     im_deconv = np.full(image.shape, 0.5, dtype=float_type)
#     psf_mirror = np.flip(psf)

#     for _ in range(iterations):
#         conv = convolve(im_deconv, psf, mode="same")
#         if filter_epsilon:
#             relative_blur = np.where(conv < filter_epsilon, 0, image / conv)
#         else:
#             relative_blur = image / conv
#         im_deconv *= convolve(relative_blur, psf_mirror, mode="same")

#         if clip:
#             # im_deconv[im_deconv > 1] = 1
#             # im_deconv[im_deconv < -1] = -1
#             im_deconv[im_deconv < 0] = 0

#     return im_deconv


# # psf_rl_0 = np.ones((5, 5,5)) / 25
# im = cropped_image[0, :, :, :]
# # im = cropped_image[0,:,:,:]

# prior_x = rl(im, psf_rl_0, iterations=30, clip=True)
# #%%
# padded_im = np.zeros_like(cropped_image[0, :, :, :])
# padded_im[grid_z_psf, grid_y_psf, grid_x_psf] = prior_x[
#     grid_z_psf, grid_y_psf, grid_x_psf
# ]
# # plt.imshow(np.max(im,axis=0))
# # plt.show()
# # fig, ax = plt.subplots(1, 3, figsize=(15, 15))
# # xyz_viewer(im)
# xyz_viewer(padded_im)
# xyz_viewer(cropped_image[0, :, :, :])
# # xyz_viewer(prior_x)
# # xyz_viewer(prior_x[60, :, :])


# # plt.imshow(np.max(cropped_image[0, :, :, :], axis=0))
# # plt.show()
# # plt.imshow(np.max(prior_x, axis=0))
# # plt.show()
# # plt.imshow(prior_x[60, :, :])
# # plt.show()
# # plt.imshow(np.sum(prior_x,axis=2))

# # from skimage.exposure import equalize_hist


# # plt.imshow(np.max(equalize_hist(prior_x), axis=0))
# # plt.show()


# # %% Two components
# # iterations = 5
# prior_x = im_deconv = np.full(cropped_image[0, :, :, :].shape, 0.5)
# # prior_x = im_deconv = np.full(image_rl_1.shape, 0.5)
# for i in range(1):

#     psf_mirror_0 = np.flip(psf_rl_0)
#     psf_mirror_1 = np.flip(psf_rl_1)

#     conv_0 = convolve(prior_x, psf_rl_0, mode="same")
#     weighted_conv_0 = np.multiply(pc_weighting_0, conv_0)

#     conv_1 = convolve(prior_x, psf_rl_1, mode="same")
#     weighted_conv_1 = np.multiply(pc_weighting_1, conv_1)
#     weighted_conv_1 = np.zeros_like(weighted_conv_0)
#     # y_0 = np.divide(y,weighted_conv_0 + weighted_conv_1)
#     # y_1 = np.divide(y,weighted_conv_0 + weighted_conv_1)

#     # unweighted_belief_0 = np.multiply(pc_weighting_0,y_0)
#     # unweighted_belief_1 = np.multiply(pc_weighting_1,y_1)

#     weighted_conv_sum = np.add(weighted_conv_0, weighted_conv_1)

#     unweighted_belief = np.true_divide(y, weighted_conv_sum)

#     weighted_belief_0 = np.multiply(unweighted_belief, pc_weighting_0)
#     weighted_belief_1 = np.multiply(unweighted_belief, pc_weighting_1)

#     # conv_0_belief = convolve(unweighted_belief_0, psf_mirror_0, mode="same")
#     # conv_1_belief = convolve(unweighted_belief_1, psf_mirror_1, mode="same")

#     conv_0_belief = convolve(weighted_belief_0, psf_mirror_0, mode="same")
#     conv_1_belief = convolve(weighted_belief_1, psf_mirror_1, mode="same")

#     conv_1_belief = np.zeros_like(conv_0_belief)

#     conv_belief = np.add(conv_0_belief, conv_1_belief)

#     psf_normaliser_0 = convolve(np.ones_like(y), psf_mirror_0, mode="same")
#     psf_normaliser_1 = convolve(np.ones_like(y), psf_mirror_1, mode="same")

#     psf_normaliser_0_weighted = np.multiply(psf_normaliser_0, pc_weighting_0)
#     psf_normaliser_1_weighted = np.multiply(psf_normaliser_1, pc_weighting_1)

#     psf_normaliser_0_weighted = 0.5
#     psf_normaliser_1_weighted = 0.5

#     # unweighted_belief = unweighted_belief_0+unweighted_belief_1
#     psf_normaliser_weighted = psf_normaliser_0_weighted + psf_normaliser_1_weighted

#     prior_x = np.multiply(np.divide(prior_x, psf_normaliser_weighted), conv_belief)

#     # prior_x[prior_x < 0] = 0
# xyz_viewer(prior_x)
# xyz_viewer(cropped_image[0, :, :, :])
# # %%


# # plt.imshow(np.max(unweighted_belief,axis=0))
# # plt.show()
# # plt.imshow(np.max(unweighted_belief,axis=1))
# # plt.show()
# # plt.imshow(np.max(unweighted_belief,axis=2))


# # plt.imshow(np.max(psf_normaliser_weighted,axis=0))
# # plt.show()
# # plt.imshow(np.max(psf_normaliser_weighted,axis=1))
# # plt.show()
# # plt.imshow(np.max(psf_normaliser_weighted,axis=2))


# #%%

# for i in range(iterations):

#     conv_1 = convolve(im_deconv, psf_rl_1, mode="same")
#     conv_2 = convolve(im_deconv, psf_rl_2, mode="same")

# for i in range(iterations):
#     conv = convolve(im_deconv, psf_rl, mode="same")
#     im_deconv = convolve(relative_blur, psf_mirror, mode="same")
# # return im_deconv

# #%%


# def richardson_lucy_varying(
#     image, eigen_psfs, psf_weights, iterations=50, clip=True, filter_epsilon=None
# ):
#     # eigen_psfs, list of N x (n_z,n_y,n_x) psfs
#     # psf_weights N x image.shape
#     assert psf_weights.shape[0:] == image.shape

#     im_deconv = np.full(image.shape, 0.5, dtype=float_type)
#     psf_mirror = np.flip(psf)

#     for i in range(iterations):
#         conv = convolve(im_deconv, psf, mode="same")
#         im_deconv = convolve(relative_blur, psf_mirror, mode="same")
#     return im_deconv


# # %%
# # def richardson_lucy(image, psf, iterations=50, clip=True, filter_epsilon=None):
# #     """Richardson-Lucy deconvolution.
# #     Parameters
# #     ----------
# #     image : ndarray
# #        Input degraded image (can be N dimensional).
# #     psf : ndarray
# #        The point spread function.
# #     iterations : int, optional
# #        Number of iterations. This parameter plays the role of
# #        regularisation.
# #     clip : boolean, optional
# #        True by default. If true, pixel value of the result above 1 or
# #        under -1 are thresholded for skimage pipeline compatibility.
# #     filter_epsilon: float, optional
# #        Value below which intermediate results become 0 to avoid division
# #        by small numbers.
# #     Returns
# #     -------
# #     im_deconv : ndarray
# #        The deconvolved image.
# #     Examples
# #     --------
# #     >>> from skimage import img_as_float, data, restoration
# #     >>> camera = img_as_float(data.camera())
# #     >>> from scipy.signal import convolve2d
# #     >>> psf = np.ones((5, 5)) / 25
# #     >>> camera = convolve2d(camera, psf, 'same')
# #     >>> camera += 0.1 * camera.std() * np.random.standard_normal(camera.shape)
# #     >>> deconvolved = restoration.richardson_lucy(camera, psf, 5)
# #     References
# #     ----------
# #     .. [1] https://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution
# #     """
# #     float_type = np.promote_types(image.dtype, np.float32)
# #     image = image.astype(float_type, copy=False)
# #     psf = psf.astype(float_type, copy=False)
# #     im_deconv = np.full(image.shape, 0.5, dtype=float_type)
# #     psf_mirror = np.flip(psf)

# #     for _ in range(iterations):
# #         conv = convolve(im_deconv, psf, mode='same')
# #         if filter_epsilon:
# #             relative_blur = np.where(conv < filter_epsilon, 0, image / conv)
# #         else:
# #             relative_blur = image / conv
# #         im_deconv *= convolve(relative_blur, psf_mirror, mode='same')

# #     if clip:
# #         im_deconv[im_deconv > 1] = 1
# #         im_deconv[im_deconv < -1] = -1

# #     return im_deconv


# # # %% Machine learning inference (is not super impressive)
# # %%


# # # https://stats.stackexchange.com/questions/229092/how-to-reverse-pca-and-reconstruct-original-variables-from-several-principal-com
# # radius_samples = np.linspace(0,1,5)
# # fig,ax = plt.subplots(nrows=2,ncols=len(radius_samples),figsize=(16,7))

# # for i,radius in enumerate(radius_samples):
# #     current_frame = np.rint(np.multiply(np.subtract(astro.shape,1),radius)).astype(int)
# #     idx = np.ravel_multi_index(current_frame,dims=astro.shape)
# #     # idx = np.ravel_multi_index(current_frame,dims=(128,128))
# #     mu = np.mean(flat_psf, axis=0)
# #     Xhat = np.dot(principle_components[:,:n_components], pca.components_[:n_components,:]) + mu
# #     # fig,ax = plt.subplots(nrows=1,ncols=len(radius_samples),figsize=(16,7))

# #     ax[0,i].imshow(psf_window_volume[idx,:,:]);ax[0,i].set_title("True | Radius: " + str(radius))
# #     ax[1,i].imshow(Xhat[idx].reshape(psf_width,psf_height)); ax[1,i].set_title("Predicted | Radius: " + str(radius))
# #     # print(current_frame)
# #     # psf_current = psf_vary(psf_window_h,psf_window_w,radius,scale)
# #     # ax[i].imshow(psf_current);ax[i].set_title("Radius: " + str(radius))
# # plt.show()
# # print(pca.explained_variance_ratio_)

# # %%
# # from sklearn.ensemble import RandomForestRegressor
# # from sklearn.neural_network import MLPRegressor
# # from sklearn.model_selection import train_test_split
# # from sklearn.gaussian_process import GaussianProcessRegressor
# # from sklearn.kernel_ridge import KernelRidge

# # # y = melt_df["Weight"].to_numpy()
# # # X = melt_df["PC"].reset_index()

# # y = melt_df["Weight"]
# # X = melt_df.drop("Weight", 1)

# # X_train, X_test, y_train, y_test = train_test_split(X, y)

# # est = RandomForestRegressor()
# # # est = MLPRegressor(max_iter=2000)
# # est = GaussianProcessRegressor()
# # # est = KernelRidge()
# # est.fit(X_train, y_train)

# # print(
# #     f"Training score: {est.score(X_train,y_train)} and Pred score: {est.score(X_test,y_test)}"
# # )
# # # %%
# # import scipy.stats as stats
# # from sklearn.utils.fixes import loguniform
# # from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# # # param_dist = {"alpha": loguniform(1e-4, 1e0)}

# # clf = est
# # clf = RandomForestRegressor()
# # param_dist = {
# #     "n_estimators": [int(x) for x in np.linspace(start=200, stop=2000, num=10)],
# #     "max_features": ["auto", "sqrt"],
# #     "max_depth": [int(x) for x in np.linspace(10, 110, num=11)],
# #     "min_samples_split": [2, 5, 10],
# #     "min_samples_leaf": [1, 2, 4],
# #     "bootstrap": [True, False],
# # }

# # # run randomized search
# # # n_iter_search = 20
# # random_search = RandomizedSearchCV(
# #     clf,
# #     param_distributions=param_dist,
# #     n_iter=100,
# #     cv=3,
# #     verbose=100,
# #     random_state=42,
# #     n_jobs=-1,
# # )

# # random_search.fit(X, y)
# # random_search.best_score_

# # # %%

# # from scipy.stats import randint, uniform

# # clf = MLPRegressor(max_iter=2000)

# # param_dist = {
# #     "hidden_layer_sizes": [
# #         (
# #             randint.rvs(100, 600, 1),
# #             randint.rvs(100, 600, 1),
# #         ),
# #         (randint.rvs(100, 600, 1),),
# #     ],
# #     "activation": ["tanh", "relu", "logistic"],
# #     "solver": ["sgd", "adam", "lbfgs"],
# #     "alpha": uniform(0.0001, 0.9),
# #     "learning_rate": ["constant", "adaptive"],
# # }

# # # run randomized search
# # # n_iter_search = 20
# # random_search = RandomizedSearchCV(
# #     clf,
# #     param_distributions=param_dist,
# #     n_iter=100,
# #     cv=3,
# #     verbose=100,
# #     random_state=42,
# #     n_jobs=-1,
# # )

# # random_search.fit(X, y)
# # random_search.best_score_


# # %%
# # print(cropped.shape)
# # # %%
# # import dask_image.imread
# # import dask_image.ndmeasure
# # import dask_image.ndfilters
# # import dask as da

# # # %%

# # # from dask.distributed import Client
# # # client = Client(processes=False, silence_logs=False)

# # image_da = dask_image.imread.imread(beads_file)


# # def scaleImageDask(image, dtype=np.uint8):
# #     scaled = image / image.max()
# #     # scaled_255 = scaled * (np.info(dtype).max)
# #     scaled_255 = scaled * 255

# #     scaled_255_8bit = scaled_255.astype(dtype)
# #     # output = scaled_255_8bit
# #     return scaled_255_8bit


# # def getPSFcoordsDask(image, window=psf_window, sigma=6):
# #     # scaled = image / image.max()

# #     # plt.imshow(eq_scaled_255_8bit.max(axis=0))

# #     img = image

# #     # Get local maximum values of desired neighborhood
# #     # I'll be looking in a 5x5x5 area
# #     img_max = dask_image.ndfilters.maximum_filter(
# #         img, size=np.divide(window, 2).astype(int)
# #     )

# #     # Threshold the image to find locations of interest
# #     # I'm assuming 6 standard deviations above the mean for the threshold
# #     img_thresh = img_max.mean() + img_max.std() * sigma

# #     # Since we're looking for maxima find areas greater than img_thresh

# #     labels, num_labels = dask_image.ndmeasure.label(img_max > img_thresh)

# #     # Get the positions of the maxima
# #     coords = dask_image.ndmeasure.center_of_mass(img, label_image=labels)

# #     #  index=da.array.arange(1, num_labels + 1)

# #     # Get the maximum value in the labels
# #     values = dask_image.ndmeasure.maximum(img, label_image=labels)
# #     return coords

# # psf_image_scaled = scaleImageDask(image_da, np.uint8)
# # # eq_scaled_255_8bit = equalize_adapthist(psf_image_scaled)
# # # eq_scaled_255_8bit = psf_image_scaled.map_blocks(equalize_adapthist, dtype=np.uint8)
# # coords = getPSFcoordsDask(psf_image_scaled, psf_window)
# # # result = coords.compute()
# # # result = client.compute(coords)
# # result = client.persist(coords)
# # # result.result()

# # result = getCoords(image_da)
# # eq_scaled_255_8bit = apply_parallel(equalize_adapthist, psf_image_scaled)
# # eq_scaled_255_8bit = equalize_adapthist(psf_image_scaled)

# # %%

# # plt.imshow(equalize_adapthist(psf_image.max(axis=0)))

# # put a blue dot at (10, 20)
# # plt.scatter(np.array(coords)[:, -1], np.array(coords)[:, -2])
# # plt.show()

# # %%
# # cropped = cropND(img, reversed(coords[0]))

# # %%
# # import dask_image.imread

# # image = dask_image.imread.imread(beads_file)

# # np_tif_stack = np.array(tif_stack)
# # %%
# # print("Loading beads")
# # # tif_stack = pims.Bioformats(beads_file, java_memory="1024m")
# # # image = tif_stack

# # print("Making numpy array")
# # # np_image = np.array(image)
# # np_image = image
# # # np_image = np.vectorise(image[0:2])
# # print("Scaling to max 1")
# # scaled = np_image / np_image.max()
# # print("Scaling to  255")
# # scaled_255 = scaled * 255
# # print("8bit")

# # scaled_255_8bit = scaled_255.astype(np.uint8)
# # output = scaled_255_8bit

# # with ProgressBar():
# #     output.compute(memory_limit="8GB")
# # %%

# # img = psf_image_scaled


# # import scipy.ndimage as ndimage
# # import dask

# # # Get local maximum values of desired neighborhood
# # # I'll be looking in a 5x5x5 area
# # img_max = dask.delayed(ndimage.maximum_filter)(img, psf_window)

# # img_thresh = img_max.mean() + img_max.std() * 6

# # label_out = dask.delayed(ndimage.label)(img_max > img_thresh)

# # labels = label_out[0]
# # num_labels = label_out[1]


# # coords = dask.delayed(ndimage.measurements.center_of_mass)(
# #     img, labels=labels, index=dask.delayed(np.arange)(1, num_labels + 1)
# # )

# # values = dask.delayed(ndimage.measurements.maximum)(
# #     img, labels=labels, index=dask.delayed(np.arange)(1, num_labels + 1)
# # )


# # with ProgressBar():
# #     raw_coords = coords.compute(memory_limit="8GB")


# # # Get the positions of the maxima
# # coords = ndimage.measurements.center_of_mass

# # values = ndimage.measurements.maximum(img, labels=labels, index=np.arange(1, num_labels + 1))

# # values.compute(memory_limit="32GB")
# # %%
# # labels, num_labels = ndimage.label(img_max > img_thresh)

# # Get the positions of the maxima
# # coords = ndimage.measurements.center_of_mass(
# #     img, labels=labels, index=np.arange(1, num_labels + 1)
# # )

# # Get the maximum value in the labels
# # values = ndimage.measurements.maximum(
# #     img, labels=labels, index=np.arange(1, num_labels + 1)
# # )

# # np.save("output/np_image_scaled01", scaled_255_8bit)
# # %%
# # print("Saving")
# # from dask.diagnostics import ProgressBar

# # with ProgressBar():
# #     output.to_zarr(
# #         "output/dask_image_scaled01.zarr", overwrite=True, memory_limit="32GB"
# #     )
# #     # output.to_hdf5("output/dask_image_scaled01.zarr", memory_limit="32GB")

# # # output.compute(memory_limit="64GB")


# # %%
# ##################print("Saving")
# # from dask.diagnostics import ProgressBar

# # with ProgressBar():
# #     output.to_zarr(
# #         "output/dask_image_scaled01.zarr", overwrite=True, memory_limit="32GB"
# #     )
# #     # output.to_hdf5("output/dask_image_scaled01.zarr", memory_limit="32GB")

# # # output.compute(memory_limit="64GB")

# # # from scipy import ndimage, misc

# # # result = ndimage.maximum_filter(image, size=np.divide(psf_window, 4).astype(int))

# # # %%
# # # max_z_filtered = result.max(0)
# # # max_z = np_tif_stack.max(0)
# # # %%
# # # %%

# # plt.imshow(max_z)
# # # %%
# # plt.imshow(max_z_filtered)
# # # %%

# # # %%
# # from skimage import data, feature

# # localisations = feature.blob_log(np_tif_stack, threshold=0.5, max_sigma=40)

# # # %%
# # fig, axes = plt.plot()

# # axes.imshow(max_z_filtered)

# # for blob in blobs:
# #     z, ay, x, r = blob
# #     c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
# #     axes.add_patch(c)
# # axes.set_axis_off()

# # plt.tight_layout()
# # plt.show()
# # %%