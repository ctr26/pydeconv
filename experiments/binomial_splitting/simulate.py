import argparse
import matplotlib
import numpy as np
from pydeconv.simulate import SimulateImagePoisson
from pydeconv import deconvolve, optics, utils
import matplotlib.pyplot as plt
import binomial_splitting as bs
from tqdm import tqdm
from pydeconv.deconvolve import RichardsonLucy

from pydeconv.metrics import ReconstructionMetrics2D, Metric
from pydeconv import metrics
import pandas as pd
import seaborn as sns
import xarray as xr


# Make more general
def simulation(
    numPixel=(256, 256),
    midPos=(128, 128),
    pxSize=(0.04, 0.04),
    lambda0=0.520,
    seed=10,
    obj_name="spokes",
    max_photons=1e2,
    na=0.8,
    coin_flip_bias=0.5,
    niter=512,
    metric_fn="kl",
):
    np.random.seed(seed)
    simSize = np.multiply(numPixel, pxSize)
    # do_image_generation = not (no_image_generation)
    # do_analysis = not (no_analysis)
    # do_save_csv = not (no_save_csv)
    # do_show_figures = not (no_show_figures)
    # do_save_images = not (no_save_images)

    # if no_show_figures:
    #     print("Don't print figures")
    #     plt.ioff()
    #     matplotlib.use("Agg")
    simulator = SimulateImagePoisson(
        obj_name=obj_name,
        numPixel=numPixel,
        midPos=midPos,
        pxSize=pxSize,
        lambda0=lambda0,
        max_photons=max_photons,
        NA=na,
    )
    obj = simulator.get_object()
    noisy_obj = simulator.simulate()
    psf, otf, fwd, bwd = simulator.get_optical_operators()

    V, T = bs.split(noisy_obj, p=coin_flip_bias, norm=True)
    objs = np.stack((obj, noisy_obj, V, T), axis=0)
    # objs = objs[0]
    # rl_steps = np.expand_dims(objs, 0).repeat(niter, axis=0)
    # rl_steps[0] = objs
    rl = RichardsonLucy(psf, max_iterations=niter, early_stopping=None)
    rl_steps = rl.deconvolve(objs, history=True)
    # TODO rl is broken

    est = rl_steps[1:]
    y_est = fwd(est)
    gt = fwd(objs)

    # recon_metrics_objs = ReconstructionMetrics2D(
    #     est=np.expand_dims(est, 1),
    #     gt=np.expand_dims(objs, 1),
    # )

    # recon_metrics_self = ReconstructionMetrics2D(
    #     est=np.expand_dims(est, 1),
    #     gt=np.expand_dims(est, 2),
    # )
    recon_metrics_gt = ReconstructionMetrics2D(
        est=np.expand_dims(y_est, 1),
        gt=np.expand_dims(gt, 1),
    )

    data_array_gt = xr.DataArray(
        recon_metrics_gt(metric_fn),
        dims=("iterations", "gt", "est"),
        coords={
            "gt": [
                "y_est_gt",
                "y_est",
                "y_est_v",
                "y_est_t",
            ],
            "est": [
                "y_gt",
                "y",
                "y_gt_v",
                "y_gt_t",
            ],
        },
    ).rename(metric_fn)
    return data_array_gt.to_dataframe()


def main(
    numPixel=(256, 256),
    midPos=(128, 128),
    pxSize=(0.04, 0.04),
    lambda0=0.520,
    seed=10,
    no_image_generation=False,
    no_analysis=False,
    no_save_csv=False,
    no_show_figures=False,
    no_save_images=False,
    # Variables,
    na=0.8,
    max_photons=1e2,
    obj_name="spokes",
    # possible objects are: 'spokes', 'points_random', 'test_target',
    niter=512,
    save_images=True,
    savefig=True,
    out_dir="results",
    coin_flip_bias=0.5,
):
    return None


def plot_score_splitting(
    numPixel=(256, 256),
    midPos=(128, 128),
    pxSize=(0.04, 0.04),
    lambda0=0.520,
    seed=10,
    # Variables,
    na=0.8,
    max_photons=1e2,
    obj_name="spokes",
    # possible objects are: 'spokes', 'points_random', 'test_target',
    niter=512,
    save_images=True,
    savefig=True,
    out_dir="results",
    coin_flip_bias=0.5,
):
    data_array_gt = simulation(
        numPixel=numPixel,
        midPos=midPos,
        pxSize=pxSize,
        lambda0=lambda0,
        seed=seed,
        obj_name=obj_name,
        max_photons=max_photons,
        na=na,
        coin_flip_bias=coin_flip_bias,
        niter=niter,
        metric_fn="kl",
    )

    df_gt_kl = data_array_gt.xs("y_est_gt", level="gt", drop_level=False).drop(
        "y_gt", level="est"
    )

    # g = sns.lineplot(
    #     data=df_gt_kl.reset_index(),
    #     hue="est",
    #     x="iterations",
    #     y="kl",
    #     # kind="line",
    # )
    # g.set(xlim=(0, 100))
    # g.set(yscale="log")
    # sns.scatterplot(
    #     data=df_gt_kl.groupby(level=("gt", "est")).apply(minima).reset_index(),
    #     hue="est",
    #     x="min_x",
    #     y="min_y",
    #     # kind="line",
    #     ax=g,
    # )

    # # g.add_legend()

    # g

    g = sns.FacetGrid(
        data=df_gt_kl.reset_index(),
        hue="est",
        # x="iterations",
        # y="kl_divergence",
        # kind="line",
    )
    g.map(sns.lineplot, "iterations", "kl")
    g.map_dataframe(facet_minima, "iterations", "kl", hue="est")

    g.set(xlim=(0, 100))
    g.set(yscale="log")
    g.add_legend()
    g.savefig("results/multiple_splitting.png")


def facet_minima(*args, **kwargs):
    x, y = args
    # hue = kwargs.pop("hue")
    data = kwargs.pop("data")
    index = list(set(data.columns) - {*args})

    minimas = data.set_index([x] + index).groupby(index).apply(minima)
    plt.scatter(minimas["min_x"], minimas["min_y"], color="red")

    return plt.gca()


def minima(df):
    min_x = df.squeeze().argmin()
    min_y = df.iloc[min_x]
    return pd.DataFrame({"min_x": min_x, "min_y": min_y})


if __name__ == "__main__":
    main()
