import numpy as np
from tqdm import tqdm
from .base import DeconvolveBase, IterativeDeconvolve
from . import stopping


class RichardsonLucy(IterativeDeconvolve):
    def __init__(
        self, psf, max_iterations=25, early_stopping=stopping.EarlyStopping()
    ):
        super().__init__(
            psf,
            max_iterations=max_iterations,
            early_stopping=early_stopping,
        )

    def est_0(self, image):
        return self.est_grey(image)

    def step(self, y, steps):
        return step(est=steps[-1], img=y, fwd=self.fwd, bwd=self.bwd)


def step(est, img, fwd, bwd):
    convEst = fwd(est)
    ratio = img / (convEst + 1e-12)
    convRatio = bwd(ratio)
    convRatio = convRatio / bwd(np.ones_like(img))
    rl_step = est * convRatio
    # plt.imshow(fwd(img[0]))
    return np.clip(rl_step, 0, np.inf)


# # Standard this as classes using pydeconv
# def richardson_lucy_history(img, fwd, bwd, niter=100):
#     iterations = np.arange(0, niter)
#     est = np.ones_like(img) * np.mean(img, axis=(1, 2), keepdims=True)
#     # Define variable that saves the reconstruction in each iteration
#     est_history = np.zeros_like(est)
#     est_history = np.repeat(est_history[np.newaxis], niter, axis=0)

#     # Richardson-Lucy deconvolution of split data
#     for l in tqdm(iterations):
#         est_history[l] = step(est_history[-1], img, fwd, bwd)

# def richarson_lucy_step(est, img, fwd, bwd):
#     convEst = fwd(est)
#     ratio = img / (convEst + 1e-12)
#     convRatio = bwd(ratio)
#     convRatio = convRatio / bwd(np.ones_like(img))
#     return est * convRatio


def richardson_lucy_history(img, fwd, bwd, iterations=25):
    history = np.empty((iterations + 1, *img.shape))
    history[0] = img
    for i in tqdm(range(iterations)):
        history[i + 1] = step(img[i], img[0], fwd, bwd)
    return history


def richardson_lucy(img, fwd, bwd, iterations=25):
    return richardson_lucy_history(img, fwd, bwd, iterations)[-1]
