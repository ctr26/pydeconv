import argparse
import matplotlib
import numpy as np
from pydeconv.simulate import SimulateImagePoisson
from pydeconv import deconvolve
import matplotlib.pyplot as plt

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

main()