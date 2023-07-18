import argparse
import matplotlib
import numpy as np
from pydeconv.simulate import SimulateImagePoisson
from pydeconv import deconvolve, optics, utils
import matplotlib.pyplot as plt
import binomial_splitting as bs
from tqdm import tqdm
from pydeconv.deconvolve import RichardsonLucy

from pydeconv.metrics import ReconstructionMetrics2D
from pydeconv import metrics


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
    objs = np.stack((noisy_obj, V, T), axis=0)
    objs = objs[0]
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
    recon_metrics = ReconstructionMetrics2D(
        x_est=rl_steps, y_est=fwd(rl_steps), x_gt=obj, y_gt=fwd(obj)
    )
    metrics_dict = metrics.get_metrics_dict(
        [
            recon_metrics.loglikelihood,
            recon_metrics.poisson,
            # recon_metrics.ncc,
            recon_metrics.crossentropy,
            recon_metrics.kl_noiseless_signal,
        ]
    )
    metrics_dict
    # kl_est_noiseless_signal = np.sum(
    #     fwd(obj) * np.log((fwd(obj) + 1e-9) / (fwd(rl_steps) + 1e-9)),
    #     axis=(-2, -1),
    # )
    rl_steps


main()
