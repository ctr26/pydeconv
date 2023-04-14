# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import minmax_scale

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
    .xs(200, level="niter", drop_level=False)
)
# %%

# %%


df[yaxis] = df[yaxis].groupby(metadata).transform(minmax_scale)


df["iteration_optimum"] = df.iloc[df.reset_index().groupby(metadata)["score"].idxmax()][
    "iterations"
]

# groups = df.groupby(level="niter")
# group = groups.get_group(list(groups.groups)[0])


def subtract_gt(df):
    # gt = df.xs("negative_kl_est_noiseless_signal", level="measurands")
    return df.groupby("measurands").apply(
        lambda x: x - df.xs("negative_kl_est_noiseless_signal", level="measurands" )
    )


baselined_df = (
    df["iteration_optimum"]
    .drop_duplicates()
    .groupby(level="max_photons")
    .apply(subtract_gt)
)
print(baselined_df)
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
na = 0.8

sns.lmplot(
    x="max_photons",
    y="iteration_optimum",
    col="obj_name",
    #    row="measurands",
    hue="measurands",
    data=df.xs(1.4, level="na").reset_index(),
    sharey=False,
    fit_reg=False,
)
plt.xscale("log")
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
