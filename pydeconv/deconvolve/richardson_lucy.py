import numpy as np


def step(est, img, fwd, bwd):
    convEst = fwd(est)
    ratio = img / (convEst + 1e-12)
    convRatio = bwd(ratio)
    convRatio = convRatio / bwd(np.ones_like(img))
    return est * convRatio


# Standard this as classes using pydeconv
def richarson_lucy(img, fwd, bwd, niter=100):
    iterations = np.arange(0, niter)
    est = np.ones_like(img) * np.mean(img, axis=(1, 2), keepdims=True)
    # Define variable that saves the reconstruction in each iteration
    est_history = np.zeros_like(est)
    est_history = np.repeat(est_history[np.newaxis], niter, axis=0)

    # Richardson-Lucy deconvolution of split data
    for l in tqdm(iterations):
        est_history[l] = step(est_history[-1], img, fwd, bwd)