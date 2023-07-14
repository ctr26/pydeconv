import argparse
import matplotlib
import numpy as np
from .simulate import image
from image import SimulateImagePoisson
from . import deconvolve

numPixel = (256, 256),
midPos = (128, 128),
pxSize = (0.04, 0.04),
n = 1.00,
lambda0 = 0.520,
seed = 10,
no_image_generation = False,
no_analysis = False,
no_save_csv = False,
no_show_figures = False,
no_save_images = False,
# Variables
na = 0.8,
max_photons = 1e2,
obj_name = "spokes",  # possible objects are: 'spokes', 'points_random', 'test_target'
niter = 500,
save_images = True,
savefig = True,
out_dir = "results",
coin_flip_bias = 0.5

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=out_dir, type=str)
    parser.add_argument("--savefig", default=savefig, type=int)
    # parser.add_argument("--save_images", default=save_images, type=int)

    # parser.add_argument("--no_image_generation", action="store_true")
    # parser.add_argument("--no_analysis", action="store_true")
    # parser.add_argument("--no_save_csv", action="store_true")
    # parser.add_argument("--no_show_figures", action="store_true")
    # parser.add_argument("--no_save_images", action="store_true")


    parser.add_argument("--niter", default=niter, type=int)
    parser.add_argument("--coin_flip_bias", default=coin_flip_bias, type=float)
    parser.add_argument("--na", default=na, type=float)
    # parser.add_argument("--max_photons", default=max_photons, type=float)
    parser.add_argument("--seed", default=seed, type=int)
    # parser.add_argument("--obj_name", default=obj_name, type=str)

    try:
        args = parser.parse_args()
    # globals().update(vars(args))
    except:
        args = parser.parse_args([])
    globals().update(vars(args))
    print(vars(args))
    # if __name__ == "__main__":
    np.random.seed(seed)
    # Path(out_dir).mkdir(parents=True, exist_ok=True)

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
    obj = SimulateImagePoisson(obj_name=obj_name,
                               numPixel=numPixel,
                                 midPos=midPos,
                                 pxSize=pxSize,
                                 lambda0=lambda0,
                                 NA=NA)