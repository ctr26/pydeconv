import numpy as np
from tqdm import tqdm
from .base import DeconvolveBase


class RichardsonLucy(DeconvolveBase):
    def __init__(self, psf, max_iterations=25, early_stopping=None):
        super().__init__(
            psf,
            max_iterations=max_iterations,
            early_stopping=early_stopping,
        )

    def est_0(self, image):
        return self.est_grey(image)

    def est_grey(self, image):
        return np.ones_like(image) + np.mean(image)

    def est_signal(self, image):
        return self.fwd(image)

    def est_half(self, image):
        return np.ones_like(image) * (image.max() / 2)

    # def deconvolution_step(self, image, i):
    #     return image

    def step(self, image, i):
        return step(est=self.steps[i], img=image, fwd=self.fwd, bwd=self.bwd)

    # def __call__(self, image):
    #     if self.history:
    #         return richardson_lucy_history(
    #             image,
    #             self.fwd,
    #             self.bwd,
    #             self.iterations,
    #         )
    #     else:
    #         return richardson_lucy(
    #             image,
    #             self.fwd,
    #             self.bwd,
    #             self.iterations,
    #         )


def step(est, img, fwd, bwd):
    convEst = fwd(est)
    ratio = img / (convEst + 1e-12)
    convRatio = bwd(ratio)
    convRatio = convRatio / bwd(np.ones_like(img))
    rl_step = est * convRatio
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
