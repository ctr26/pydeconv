import numpy as np
from tqdm import tqdm
from pydeconv import deconvolve





def split_richarson_lucy(img, fwd, bwd, niter=100):
    iterations = np.arange(0, niter)
    est_split = np.ones_like(img) * np.mean(img, axis=(1, 2), keepdims=True)
    # Define variable that saves the reconstruction in each iteration
    est_split_history = np.zeros_like(est_split)
    est_split_history = np.repeat(est_split_history[np.newaxis], niter, axis=0)

    # Richardson-Lucy deconvolution of split data
    for l in tqdm(iterations):
        est_split_history[l] = deconvolve.richarson_lucy_step(
            img, est_split_history[-1], fwd, bwd
        )
