
from scipy.interpolate import Rbf
import dask.array as da
from dask.diagnostics import ProgressBar
import pandas as pd
cropped = None
coord_list = None
flat_df = None

# def getEigenPSF(principle_component):
#     eigen_psfs = pca.components_.reshape((-1, *psf_window))
#     return eigen_psf

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

def get_pc_weighting_fun_3D(pca_df,function="cubic"):


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

def get_pc_weighting_fun_3D(pca_df,pc_component,basis_function="cubic",chunks=5):
    x_list = pca_df.index.get_level_values("x")
    y_list = pca_df.index.get_level_values("y")
    z_list = pca_df.index.get_level_values("z")

    grid_z_psf, grid_y_psf, grid_x_psf = np.mgrid[
    min(z_range_psf) : max(z_range_psf),
    min(y_range_psf) : max(y_range_psf),
    min(x_range_psf) : max(x_range_psf),
    ]

    rbi_fun = get_pc_weighting_fun_3D(pca_df,function="cubic")

    dask_xyzp = da.from_array((grid_z, grid_y, grid_x), chunks=chunks)

    with ProgressBar():
        f = da.map_blocks(
            rbfi, *dask_xyzp
        )
    # g = client.persist(f)
    pc_component_weight_map = f.compute()

    return weighting_function

def interpolate_pc_weighting(principle_component,coords):
    weighting = None
    return weighting

def getPSFdf():
    # flat = pd.DataFrame(cropped.reshape(cropped.shape[0], -1)).dropna(0).set_index(index)
    return pd.DataFrame(cropped.reshape(cropped.shape[0], -1)).dropna(0).set_index(index)

# def getPSFdf():
#     # flat = pd.DataFrame(cropped.reshape(cropped.shape[0], -1)).dropna(0).set_index(index)
#     return flat

def __main__(psf_image,psf_centres, noramlise=True, centre_crop=True):
    '''
    Function for giving in a 
    '''
    eigen_psf = None # Is a list of coord.shape arrays
    weighting = None # Weighting is a function that takes *coords
    return eigen_psf,weighting

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