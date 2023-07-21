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
    niter=500,
    save_images=True,
    savefig=True,
    out_dir="results",
    coin_flip_bias=0.5,
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

    V, T = bs.split(noisy_obj, p=coin_flip_bias)
    objs = np.stack((obj, noisy_obj, V, T), axis=0)
    # objs = objs[0]
    # rl_steps = np.expand_dims(objs, 0).repeat(niter, axis=0)
    # rl_steps[0] = objs
    rl = RichardsonLucy(psf, max_iterations=niter, early_stopping=None)
    rl_steps = rl.deconvolve(objs, history=True)
    # TODO rl is broken

    # iterations = np.arange(0, niter)

    # # Richardson-Lucy deconvolution of split data
    # image = fwd(objs)
    # est = np.ones_like(image) + np.mean(objs[0])
    # rl_steps = np.zeros_like(est)
    # rl_steps = np.repeat(rl_steps[np.newaxis], niter, axis=0)
    # for l in tqdm(iterations, desc=f"Richardson lucy deconvolution"):
    #     convEst = fwd(est)
    #     ratio = image / (convEst + 1e-12)
    #     convRatio = bwd(ratio)
    #     convRatio = convRatio / bwd(np.ones_like(image))
    #     est = est * convRatio

    #     rl_steps[l] = est

    # Richardson-Lucy deconvolution of split data
    # for i, data in tqdm(enumerate(rl_steps[:-1])):
    #     rl_steps[i] = deconvolve.richardson_lucy.step(
    #         rl_steps[i - 1], obj, simulator.fwd, simulator.bwd
    #     )
    #     # bs_steps[i] = bs.binomial_splitting_step(rl_steps[i-1], obj, simulator.fwd, simulator.bwd, coin_flip_bias=coin_flip_bias)
    # recon_metrics = ReconstructionMetrics2D(
    #     est=rl_steps,gt=fwd(obj),
    # )

    # est = np.expand_dims(rl_steps,1)
    # gt = fwd(objs)

    # recon_metrics_y = ReconstructionMetrics2D(
    #     est=fwd(np.expand_dims(rl_steps, 2)),
    #     gt=fwd(objs),
    # )

    # recon_metrics_x = ReconstructionMetrics2D(
    #     est=fwd(np.expand_dims(rl_steps, 2)),
    #     gt=objs,
    # )
    
    # y = noisy_obj
    est = fwd(rl_steps)
    gt = fwd(objs)
    
    recon_metrics_gt = ReconstructionMetrics2D(
        est=np.expand_dims(est, 1),
        gt=np.expand_dims(objs, 1),
    )


    recon_metrics_self = ReconstructionMetrics2D(
        est=np.expand_dims(est, 1),
        gt=np.expand_dims(est, 2),
    )
    # recon_metrics_y = ReconstructionMetrics2D(
    #     est=np.expand_dims(fwd(rl_steps)[:,0], 1),
    #     gt=np.expand_dims(fwd(objs)[0], 1),
    # )

    # recon_metrics = ReconstructionMetrics2D(
    #     est=rl_steps,gt=fwd(obj),
    # )
    # metrics_dict["KLSelf"] = recon_metrics_self.kl()
    
    # metrics_dict = metrics.get_metrics_dict(
    #     [
    #         recon_metrics_gt.kl,
    #         recon_metrics_self.kl,
    #     ]
    # )
    data_array_gt = xr.DataArray(
        recon_metrics_gt.kl(),
        dims=("iterations", "splitting_0", "splitting_1"),
        coords={
            "splitting_1": ["y_gt","y", "y_v", "y_t"],
            "splitting_0": ["y_gt","y", "y_v", "y_t"],
        },
    ).rename("score")
    
    g = sns.relplot(
        data=data_array_gt.to_dataframe(),
        x="iterations",
        y="score",
        row="splitting_0",
        hue="splitting_1",
        kind="line",
        facet_kws={"sharey": False, "sharex": True},
    )
    g.set(xlim=(1, 100))
    g.set(yscale="log")
    g

    # data_array_self = xr.DataArray(
    #     recon_metrics_self.kl(),
    #     dims=("iterations", "splitting_0", "splitting_1"),
    #     coords={
    #         "splitting_1": ["y", "y_v", "y_T"],
    #         "splitting_0": ["y", "y_v", "y_T"],
    #     },
    # )
    
    # g = sns.relplot(
    #     data=data_array_self.rename("score").to_dataframe(),
    #     x="iterations",
    #     y="score",
    #     row="splitting_0",
    #     col="splitting_1",
    #     kind="line",
    #     facet_kws={"sharey": False, "sharex": True},
    # )
    # g.set(xlim=(1, 500))
    # g.set(yscale="log")
    # g


    # # metrics_dict
    # # kl_est_noiseless_signal = np.sum(
    # #     fwd(obj) * np.log((fwd(obj) + 1e-9) / (fwd(rl_steps) + 1e-9)),
    # #     axis=(-2, -1),
    # # )
    # # rl_steps
    # # for metrics_fn in metrics_dict:
    # #     # print(f"{metrics_fn.__name__}: {metrics_dict[metrics_fn]}")
    # #     for obj in metrics_dict[metrics_fn]:
    # #         print(obj)
    # #         # plt.figure()
    # #         # plt.plot(obj)
    # #         # plt.show()

    # # # Create an empty list to store temporary dataframes
    # # dfs = []

    # # # For each key and value in the dictionary
    # # for key, value in metrics_dict.items():
    # #     # Create a DataFrame from the value
    # #     value_df = pd.DataFrame(value.T, index=['obj', 'V', 'T']).rename_axis('splitting')

    # #     # Add a 'dict_key' column with the key value
    # #     value_df['dict_key'] = key

    # #     # Melt the DataFrame to long format and append it to the list
    # #     dfs.append(value_df)

    # # # Concatenate all the temporary dataframes
    # # df_long = pd.concat(dfs).set_index('dict_key',append=True)
    # recon_metrics_y = ReconstructionMetrics2D(
    #     est=np.expand_dims(est[:,0], 1),
    #     gt=np.expand_dims(gt[0], 1),
    # )
    
    # df = pd.concat(
    #     {
    #         k: pd.DataFrame(v.T, index=["obj", "V", "T"])
    #         for k, v in metrics_dict.items()
    #     },
    #     names=["dict_key", "splitting"],
    # )  #

    # df_long = df.reset_index().melt(
    #     id_vars=["dict_key", "splitting"], var_name="iteration", value_name="score"
    # )

    # g = sns.relplot(
    #     data=df_long,
    #     x="iteration",
    #     y="score",
    #     row="splitting",
    #     col="dict_key",
    #     hue="splitting",
    #     style="dict_key",
    #     kind="line",
    #     facet_kws={"sharey": False, "sharex": True},
    # )
    # g.set(xlim=(0, 100))
    # g.set(yscale="log")
    # g


main()
