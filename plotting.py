# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import minmax_scale
import dask.dataframe as dd
from dask.diagnostics import ProgressBar

# ddf = dd.read_csv("results/**/*.csv").compute().to_csv("results/full.csv")

# with ProgressBar():
#     df = dd.read_csv("results/**/*.csv").compute()
#     df.to_csv("results/full.csv",index=False)

csv_file = "results/full.csv"
metadata = [
    "measurands",
    "seed",
    "na",
    "max_photons",
    "obj_name",
    "niter",
    "coin_flip_bias",
]

xaxis = "iterations"
yaxis = "score"

df = (
    pd.read_csv(csv_file)
    .drop("out_dir", axis=1)
    .set_index(metadata)
    .drop(["LogLikelihood", "NCCLoss", "CrossEntropyLoss"], level="measurands")
    # .xs(200, level="niter", drop_level=False)
)


# %%

# %%


df[yaxis] = df[yaxis].groupby(metadata).transform(minmax_scale)


df["iteration_optimum"] = df.iloc[df.reset_index().groupby(metadata)["score"].idxmax()][
    "iterations"
]

# groups = df.groupby(level="niter")
# group = groups.get_group(list(groups.groups)[0])

# def subtract_gt(df):
#     try:
#         gt = df.xs("negative_kl_est_noiseless_signal", level="measurands")
#         return df.groupby("measurands").apply(lambda x: x - gt)
#     except:
#         return df.groupby("measurands").apply(lambda x: np.nan)


def subtract_gt(df):
    try:
        # gt = df.xs("negative_kl_est_noiseless_signal", level="measurands", drop_level=False)
        gt = df.xs("negative_kl_est_noiseless_signal", level="measurands")
        # result = df.groupby("measurands").apply(lambda x: x - gt)
        # result = df.sub(gt, level="measurands")
        # result = df.groupby("measurands").apply(lambda x: x - gt)
        # gt = np.nan
    except:
        # result = df.copy()
        gt = np.nan
    return df - gt
    return df.groupby("measurands").apply(lambda x: x - gt)


# def subtract_gt(df):
#     # gt = df.xs("negative_kl_est_noiseless_signal", level="measurands")
#     return df.groupby("measurands").apply(
#         try:
#             lambda x: x - df.xs("negative_kl_est_noiseless_signal", level="measurands" )
#         except:
#             lambda x: x=np.nan
#     )


# df["subs_iteration_optimum"] = (
#     df[["iteration_optimum"]]
#     .drop_duplicates()
#     .subtract(
#         df[["iteration_optimum"]].xs(
#             "negative_kl_est_noiseless_signal", level="measurands"
#         )
#     )
#     # .groupby(level="max_photons")
#     # .apply(subtract_gt)
# )

# df["div_iteration_optimum"] = (
#     df["iteration_optimum"]
#     .drop_duplicates()
#     .divide(
#         df["iteration_optimum"].xs(
#             "negative_kl_est_noiseless_signal", level="measurands"
#         )
#     )
# )

# print(baselined_df)
df_optimum = df[["iteration_optimum"]].drop_duplicates()
gt = df_optimum.xs("negative_kl_est_noiseless_signal", level="measurands")
gt = gt.groupby(gt.index.droplevel('seed').names).mean()
# df = baselined_df
# groups = df["iteration_optimum"].drop_duplicates().groupby(level="max_photons")
# group = groups.get_group(list(groups.groups)[0])
# .groupby(level=["niter","measurands"])

# df.groupby(level="niter") - df.groupby(level="niter").xs("negative_kl_est_noiseless_signal", level="measurands")


# sub_groups = group.groupby("measurands")
# sub_group = sub_groups.get_group(list(sub_groups.groups)[0])

# df["iteration_optimum"].groupby(level="niter").apply(
#     lambda x: x - x.xs("negative_kl_est_noiseless_signal", level="measurands")
# )
# df = df[df["iterations"] < 250]
na = 0.7842105263157894
diff = (df_optimum - gt).droplevel("seed").assign(baseline="sub")
div = (df_optimum / gt).droplevel("seed").assign(baseline="div")

both = pd.concat([diff,div])
sns.lmplot(
    x="max_photons",
    y="iteration_optimum",
    col="obj_name",
    row="baseline",
    hue="measurands",
    x_bins=1000,
    data=(both).xs(na, level="na").reset_index(),
    sharey=False,
    fit_reg=False,
)
# plt.xscale("log")
plt.savefig(f"figures/na=({na:.2f})_vary.pdf")
plt.show()


max_photons = df.reset_index().apply(lambda col: col.unique())["max_photons"][6]

sns.lmplot(
    x="na",
    y="iteration_optimum",
    col="obj_name",
    #    row="measurands",
    hue="measurands",
    data=df.xs(max_photons, level="max_photons").reset_index(),
    sharey=False,
    fit_reg=False,
)
plt.xscale("log")
plt.savefig(f"figures/max_photons=({max_photons:.2f})_vary.pdf")
plt.show()
# %%
# sns.lmplot(
#     x="iterations",
#     y="score",
#     col="obj_name",
#     row="measurands",
#     hue="max_photons",
#     data=df.reset_index(),
#     sharey=False,
#     fit_reg=False,
# )
# plt.show()


# print("OK")
# %%


plt.show()
sns.lmplot(
    x="iterations",
    y="score",
    col="obj_name",
    row="measurands",
    hue="max_photons",
    data=df.reset_index(),
    sharey=False,
    fit_reg=False,
)
plt.show()
print("done")


sns.lmplot(
    y="iteration_optimum",
    x="score",
    col="obj_name",
    row="measurands",
    hue="max_photons",
    data=df.reset_index(),
    sharey=False,
    fit_reg=False,
)
plt.show()
print("done")