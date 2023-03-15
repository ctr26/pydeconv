import pandas as pd
from pathlib import Path
import dask.dataframe as dd
# %%
import pandas as pd
import os
import glob as glob
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import dask.dataframe as dd
# configfile: "config.yaml"
import numpy as np
from dask.diagnostics import ProgressBar
import argparse

out_dir = "results"

parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", default=out_dir, type=str)

try:
    args = parser.parse_args()
# globals().update(vars(args))
except:
    args = parser.parse_args([])
globals().update(vars(args))
print(vars(args))
# if __name__ == "__main__":

Path(out_dir).mkdir(parents=True, exist_ok=True)

glob_pattern = os.path.join(out_dir, "./**.csv")

df = dd.read_csv(glob_pattern)
print(df)
with ProgressBar():
    df = df.compute()
    df.to_csv(f"{out_dir}/full.csv",index=False)


# folder = "results_full"
# file = f"{folder}.csv"
# %%

# df = pd.read_csv(file)

# %%
# THINNING_TYPE = "poisson"
# SIGNAL_STRENGTH = 1.0
# COIN_FLIP_BIAS = 0.5
# SAVEFIG = 1
# SAVE_IMAGES = 0
# IMAGE_SCALE = 4
# PSF_SCALE = 0.5
# PSF_SIZE = 64

# index = ["psf_type", "psf_scale", "thinning_type", "coin_flip_bias", "signal_strength"]
# metadata_index = ["savefig", "save_images", "out_dir", "max_iter", "image_scale"]
# x_axis = ["iterations"]
# # df_dropped = df.drop(data_index, axis=1)
# df_clean = df.iloc[:, 2:][BEST_METRICS+index+x_axis]
# df_indexed = df_clean.set_index(index)
# # df_full = df_indexed.drop(metadata_index, axis=1)
# variables = df_indexed.columns

# df_melt = pd.melt(df_indexed, id_vars=["iterations"],
#         var_name="metric",
#         ignore_index=False).set_index(["metric"],
#         append=True)

# df_melt_slim = df_melt

# df_melt_slim = df_melt.xs(
#     (PSF_SCALE,COIN_FLIP_BIAS),
#     level=("psf_scale","coin_flip_bias"),
# )
# df_melt_slim

# # df_melt_slim.xs("log_liklihood_V",level="metric").plot()
# from sklearn.preprocessing import minmax_scale

# # df_melt_slim["normalised_value"] =
# # vars_list = list(df_melt_slim.index.names)
# # vars_list.remove("iterations");vars_list
# # %%
# ddf = dd.from_pandas(df_melt_slim.reset_index(), npartitions=32)
# groups = (
#     ddf.groupby(vars_list)
#     # .apply(minmax_scale,axis=0)
# )
#  %%
