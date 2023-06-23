# %% Import required modules
import os
import numpy as np
from scipy import special, fft, interpolate
import argparse
from PIL import Image, ImageOps
from pathlib import Path
import glob
import pandas as pd
import logging

import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib

from tqdm import tqdm
import scipy

font = {"size": 12}
rc("font", **font)
from skimage import io, util

# Do you want to save Figures as .png file?
doPrint = True

# Setting seed to make simulation reproducible

logger = logging.getLogger("dev")
logger.setLevel(logging.INFO)
# %% ---------------------------------------------------------------------------
# %% Some functions that are required for the code.


def xx(numPixel, simSize):
    x = np.linspace(-simSize[0] / 2, simSize[0] / 2, numPixel[0])
    y = np.linspace(-simSize[1] / 2, simSize[1] / 2, numPixel[1])
    xx = np.meshgrid(x, y)[0]
    return xx


def yy(numPixel, simSize):
    x = np.linspace(-simSize[0] / 2, simSize[0] / 2, numPixel[0])
    y = np.linspace(-simSize[1] / 2, simSize[1] / 2, numPixel[1])
    yy = np.meshgrid(x, y)[1]
    return yy


def rr(numPixel, simSize):
    x = xx(numPixel, simSize)
    y = yy(numPixel, simSize)
    r = np.sqrt(x**2 + y**2)
    return r


def phiphi(numPixel):
    x = xx(numPixel, numPixel)
    y = yy(numPixel, numPixel)

    phi = np.arctan2(y, x)
    return phi


def jincPSF(numPixel, midPos, pxSize, lambda0, NA):
    """Assumes isotropic number of pixels (e.g. 256 x 256"""
    lambda0 = lambda0 / pxSize[0]
    abbelimit = lambda0 / NA
    ftradius = numPixel[0] / abbelimit
    scales = ftradius / numPixel[0]

    x = xx(numPixel, numPixel)
    y = yy(numPixel, numPixel)

    r_scaled = np.pi * np.sqrt((x * scales) ** 2 + (y * scales) ** 2)

    apsf = special.jv(1, 2 * r_scaled) / (r_scaled + 1e-12)
    apsf[midPos[0], midPos[1]] = 1.0

    return apsf


def ft2d(data_in):
    mySize = np.ndim(data_in)
    myAxes = np.linspace(mySize - 1 - 1, mySize - 1, 2, dtype="int64")
    temp = fft.ifftshift(data_in, axes=myAxes)
    temp = fft.fftn(temp, axes=myAxes, norm="ortho")
    data_out = fft.fftshift(temp, axes=myAxes)

    return data_out


def ift2d(data_in):
    mySize = np.ndim(data_in)
    myAxes = np.linspace(mySize - 1 - 1, mySize - 1, 2, dtype="int64")
    temp = fft.ifftshift(data_in, axes=myAxes)
    temp = fft.ifftn(temp, axes=myAxes, norm="ortho")
    data_out = fft.fftshift(temp, axes=myAxes)
    return data_out


def abssqr(x):
    return np.abs(x) ** 2


def AbbeRadiusFromPupil(pupil, midPos):
    arr = np.argwhere(abssqr(pupil / np.max(pupil)) > 1e-3)
    R = np.max(np.sqrt((arr[:, 0] - midPos[0]) ** 2 + (arr[:, 1] - midPos[1]) ** 2))
    return np.ceil(R)


def CreateObject(obj_name, numPixel, midPos):
    x = xx(numPixel, numPixel)
    x = yy(numPixel, numPixel)
    r = rr(numPixel, numPixel)
    phi = phiphi(numPixel)

    obj = np.zeros(numPixel)
    if obj_name == "spokes":
        obj[r < (0.5 * np.max(r))] = 1.0
        obj[np.mod(phi + np.pi, 2 * np.pi / 18) < 2 * np.pi / 18 / 2] = 0.0
    elif obj_name == "test_target":
        N = numPixel[0]
        x = np.arange(N) / N - 0.5
        y = np.arange(N) / N - 0.5
        aa = np.zeros((N, N))
        aa[:: int(N / 24), :] = 1

        X, Y = np.meshgrid(x, y)
        aa *= Y + 0.5
        R = np.sqrt(X**2 + Y**2)
        f0 = 4
        k = 50
        a = 0.5 * (1 + np.cos(np.pi * 2 * (f0 * R + k * R**2 / 2)))
        a[: int(N / 2), :][R[: int(N / 2), :] < 0.4] = 0
        a[: int(N / 2), :][R[: int(N / 2), :] < 0.3] = 1
        a[: int(N / 2), :][R[: int(N / 2), :] < 0.15] = 0

        a[int(N * 3 / 4) :, int(N * 3 / 4) :] = 0
        for l in np.arange(0, 2 * k):
            ind = np.random.randint(int(N * 1 / 4) - 1, size=(2))
            d = np.random.randint(2)
            a[
                ind[0] + int(N * 3 / 4) - d + 1 : ind[0] + int(N * 3 / 4) + d + 1,
                ind[1] + int(N * 3 / 4) - d + 1 : ind[1] + int(N * 3 / 4) + d + 1,
            ] = 1
        aa[:, int(N / 4) :] = a[:, int(N / 4) :]
        aa[: int(N / 32), :] = 0
        aa[N - int(N / 48) :, :] = 0
        aa[:, : int(N / 48)] = 0
        aa[:, N - int(N / 32) :] = 0
        obj = aa
    elif obj_name == "points_random":
        pos = np.random.randint(
            midPos - np.floor(np.min(numPixel[0]) / 3).astype(int),
            midPos + np.floor(np.min(numPixel[1]) / 3).astype(int),
            size=(100, 2),
        )
        obj[pos[:, 0], pos[:, 1]] = np.random.rand(100) + 1.0

    return obj


def cat(data_in):
    return np.stack(data_in, axis=0)


def radialmean(data, center, nbins):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    r = r.astype(np.int32)

    tbin = np.bincount(r.ravel(), data.ravel())
    if True:
        X = np.arange(0, np.shape(tbin)[0])
        Xnew = np.arange(0, nbins)
        f = interpolate.interp1d(X, tbin, kind="cubic")
        tbin = f(Xnew)

    nr = np.bincount(r.ravel())
    if True:
        f = interpolate.interp1d(X, nr, kind="cubic")
        nr = f(Xnew)

    radialprofile = tbin / nr

    return radialprofile


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# %% Define simulation parameters

numPixel = (256, 256)
midPos = (128, 128)
pxSize = (0.04, 0.04)
simSize = np.multiply(numPixel, pxSize)
n = 1.00
lambda0 = 0.520
seed = 10

no_image_generation = False
no_analysis = False
no_save_csv = False
no_show_figures = False
no_save_images = False
# Variables
na = 0.8
max_photons = 1e2
obj_name = "spokes"  # possible objects are: 'spokes', 'points_random', 'test_target'
niter = 500
save_images = True
savefig = True

out_dir = "results"
coin_flip_bias = 0.5

parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", default=out_dir, type=str)
parser.add_argument("--savefig", default=savefig, type=int)
# parser.add_argument("--save_images", default=save_images, type=int)

parser.add_argument("--no_image_generation", action="store_true")
parser.add_argument("--no_analysis", action="store_true")
parser.add_argument("--no_save_csv", action="store_true")
parser.add_argument("--no_show_figures", action="store_true")
parser.add_argument("--no_save_images", action="store_true")


parser.add_argument("--niter", default=niter, type=int)
parser.add_argument("--coin_flip_bias", default=coin_flip_bias, type=float)
parser.add_argument("--na", default=na, type=float)
parser.add_argument("--max_photons", default=max_photons, type=float)
parser.add_argument("--seed", default=seed, type=int)
parser.add_argument("--obj_name", default=obj_name, type=str)

try:
    args = parser.parse_args()
# globals().update(vars(args))
except:
    args = parser.parse_args([])
globals().update(vars(args))
print(vars(args))
# if __name__ == "__main__":
np.random.seed(seed)
Path(out_dir).mkdir(parents=True, exist_ok=True)

do_image_generation = not (no_image_generation)
do_analysis = not (no_analysis)
do_save_csv = not (no_save_csv)
do_show_figures = not (no_show_figures)
do_save_images = not (no_save_images)

if no_show_figures:
    print("Don't print figures")
    plt.ioff()
    matplotlib.use("Agg")
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# %% Compute PSF, object and image


# The point-spread-function
apsf = jincPSF(numPixel, midPos, pxSize, lambda0, na)
pupil = ft2d(apsf)
kAbbe = AbbeRadiusFromPupil(pupil, midPos)
psf = abssqr(apsf)
psf = psf / np.sum(psf) * np.sqrt(np.size(psf))

# Generate groundtruth object
obj = CreateObject(obj_name, numPixel, midPos)
obj = obj / np.max(obj) * max_photons

# Forward and backwards model
otf = ft2d(psf)
fwd = lambda x: np.real(ift2d(ft2d(x) * otf))
bwd = lambda x: np.real(ift2d(ft2d(x) * np.conj(otf)))

# Obtain image information
img = fwd(obj)

# Apply shot noise
img = np.random.poisson(img).astype("int32")

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# %% Perform binomial splitting

img_T = np.random.binomial(img, 0.5)
img_V = img - img_T

# Pack into a nice format and remove
img_split = cat((img_T, img_V))
iterations = np.arange(0, niter)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
if do_image_generation:
    # niter = 200
    # Choose initialization: 1) average value as constant; 2) further blurred image
    est_split = np.ones_like(img_split) * np.mean(
        img_split, axis=(1, 2), keepdims=True
    )  # 1)
    # est_split = fwd(img_split)																# 2)

    # Define variable that saves the reconstruction in each iteration
    est_split_history = np.zeros_like(est_split)
    est_split_history = np.repeat(est_split_history[np.newaxis], niter, axis=0)

    # Richardson-Lucy deconvolution of split data
    for l in tqdm(iterations, desc=f"Richardson lucy deconvolution"):
        convEst = fwd(est_split)
        ratio = img_split / (convEst + 1e-12)
        convRatio = bwd(ratio)
        convRatio = convRatio / bwd(np.ones_like(img))
        est_split = est_split * convRatio

        est_split_history[l] = est_split

    # Reshapeing size of img split for later
    img_split = np.transpose(
        np.repeat(img_split[:, np.newaxis, :, :], niter, axis=1), axes=(1, 0, 2, 3)
    )

    # # Choose initialization: 1) average value as constant; 2) further blurred image
    est = np.ones_like(img) + np.mean(img)  # 1)
    # est = fwd(img)																		# 2)

    # Define variable that saves the reconstruction in each iteration
    est_history = np.zeros_like(est)
    est_history = np.repeat(est_history[np.newaxis], niter, axis=0)

    # Richardson-Lucy deconvolution of non-split data
    for l in tqdm(iterations):
        convEst = fwd(est)
        ratio = img / (convEst + 1e-12)
        convRatio = bwd(ratio)
        convRatio = convRatio / bwd(np.ones_like(img))
        est = est * convRatio

        est_history[l, :, :] = est
# %%
if do_image_generation:
    if do_save_images:
        # directory = os.path.join(out_dir,"est_history")
        def save_images(image_array, directory=""):
            for iteration, image in enumerate(
                tqdm(image_array, desc=f"Saving images {directory}")
            ):
                Path(directory).mkdir(parents=True, exist_ok=True)
                io.imsave(os.path.join(directory, f"{iteration:05}.tif"), image)
                # io.imsave(os.path.join(directory,f"{iteration:05}.png"),image)
                try:
                    # Image.fromarray(image).convert("L").save(os.path.join(directory,f"{iteration:05}.png"))
                    io.imsave(
                        os.path.join(directory, f"{iteration:05}.png"),
                        image.astype(np.uint8),
                    )
                except:
                    None

        save_images(est_history, os.path.join(out_dir, "est_history"))
        save_images(est_split_history, os.path.join(out_dir, "est_split_history"))

if not (do_image_generation):
    if do_save_images:
        # directory = os.path.join(out_dir,"est_history")
        def load_images(directory=""):
            path = os.path.join(directory, "*.tif")
            files = glob.glob(path)
            img_list = []
            for iteration, file in enumerate(
                tqdm(files, desc=f"Loading images {directory}")
            ):
                # img = np.array(Image.open(file))
                img = io.imread(file)
                img_list.append(img)
            return np.array(img_list)

        est_history = load_images(os.path.join(out_dir, "est_history"))
        est_split_history = load_images(os.path.join(out_dir, "est_split_history"))

    # for l in est_history
    # path = os.path.join(est_history,"est_history",iteration)
    # save_images(est_split_history,out_dir)
# if not(not(do_image_generation)):
# est_history = load_images(out_dir)
# est_split_history = load_images(out_dir)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# %% Calculate and display different criteria
if do_analysis:
    logger.info("Analysis")
    LogLikelihood = np.sum(
        -fwd(est_split_history)
        + img_split[:, ::+1] * np.log(fwd(est_split_history) + 1e-12),
        axis=(1, 2, 3),
    )
    PoissonLoss = np.sum(
        -fwd(est_split_history)
        + img_split[:, ::-1] * np.log(fwd(est_split_history) + 1e-12),
        axis=(1, 2, 3),
    )
    NCCLoss = np.squeeze(
        np.mean(
            (obj - np.mean(obj))
            * (est_history - np.mean(est_history, axis=(1, 2), keepdims=True)),
            axis=(1, 2),
            keepdims=True,
        )
        / (np.std(obj) * np.std(est_history, axis=(1, 2), keepdims=True))
    )
    CrossEntropyLoss = np.sum(
        est_split_history[:, ::-1] * np.log(est_split_history[:, ::+1] + 1e-12),
        axis=(1, 2, 3),
    )

    # CrossEntropy and PoissonLoss of not ground truth

    # kl_est_noiseless_signal = np.sum(
    #     np.expand_dims(fwd(obj), 0)
    #     * np.log((np.expand_dims(fwd(obj), 0) + 1e-9) / (fwd(est_history) + 1e-9)),
    #     axis=(1, 2),
    # )

    # The following should be the ground truth best kl divergence of

    kl_est_noiseless_signal = np.sum(
        fwd(obj) * np.log((fwd(obj) + 1e-9) / (fwd(est_history) + 1e-9)),
        axis=(1, 2),
    )

    #
    # kl_est_noiseless_signal = scipy.stats.entropy(
    #     np.expand_dims(fwd(obj), 0), fwd(est_history), axis=(1, 2)
    # )

    if do_save_csv:
        data_dict = {
            "LogLikelihood": LogLikelihood,
            "PoissonLoss": PoissonLoss,
            "NCCLoss": NCCLoss,
            "CrossEntropyLoss": CrossEntropyLoss,
            "iterations": iterations,
            "negative_kl_est_noiseless_signal": -kl_est_noiseless_signal,
        }
        metadata_dict = {
            "seed": seed,
            "na": na,
            "max_photons": max_photons,
            "obj_name": obj_name,
            "niter": niter,
            "out_dir": out_dir,
            "coin_flip_bias": coin_flip_bias,
        }
        data = pd.melt(
            pd.DataFrame(data_dict),
            id_vars=["iterations"],
            value_name="score",
            var_name="measurands",
        )
        metdata = pd.DataFrame(metadata_dict, index=data.index)
        data = data.join(metdata)
        data.to_csv(os.path.join(out_dir, "data.csv"), index=False)

    # if do_show_figures:
    plt.figure()
    ax = plt.subplot(2, 2, 1)
    plt.plot(LogLikelihood, "g-")
    plt.grid("on")
    plt.gca().axes.yaxis.set_ticks([])
    plt.xlabel("Iteration number", fontsize=10)
    plt.ylabel("Log. likelihood", fontsize=10)
    plt.title("Without binomial splitting", fontsize=8)
    plt.xlim(0, niter)
    ax = plt.subplot(2, 2, 2)
    plt.plot(PoissonLoss, "k-")
    plt.grid("on")
    plt.gca().axes.yaxis.set_ticks([])
    plt.xlabel("Iteration number", fontsize=10)
    plt.ylabel("Log. likelihood", fontsize=10)
    plt.title("With binomial splitting", fontsize=8)
    plt.xlim(0, niter)
    plt.subplot(2, 2, 3, sharex=ax)
    plt.plot(CrossEntropyLoss, "b-")
    plt.xlabel("Iteration number", fontsize=10)
    plt.ylabel("Cross entropy", fontsize=10)
    plt.grid("on")
    plt.gca().axes.yaxis.set_ticks([])
    plt.title("With binomial splitting", fontsize=8)
    plt.xlim(0, niter)
    plt.subplot(2, 2, 4, sharex=ax)
    plt.plot(NCCLoss, "r-")
    plt.xlabel("Iteration number", fontsize=10)
    plt.ylabel("Cross-correlation", fontsize=10)
    plt.gca().axes.yaxis.set_ticks([])
    plt.grid("on")
    plt.title("Knowing groundtruth object", fontsize=8)
    plt.xlim(0, niter)
    plt.tight_layout()

    max_LogLikelihood = np.argmax(LogLikelihood)
    max_PoissonLoss = np.argmax(PoissonLoss)
    max_NCCLoss = np.argmax(NCCLoss)
    max_CrossEntropyLoss = np.argmax(CrossEntropyLoss)
    max_kl_est_noiseless_signal = np.argmax(kl_est_noiseless_signal)

    plt.subplot(2, 2, 1)
    plt.plot(max_LogLikelihood, LogLikelihood[max_LogLikelihood], "go", mfc="none")
    plt.subplot(2, 2, 2)
    plt.plot(max_PoissonLoss, PoissonLoss[max_PoissonLoss], "ko", mfc="none")
    plt.subplot(2, 2, 3)
    plt.plot(
        max_CrossEntropyLoss, CrossEntropyLoss[max_CrossEntropyLoss], "bo", mfc="none"
    )
    plt.subplot(2, 2, 4)
    plt.plot(max_NCCLoss, NCCLoss[max_NCCLoss], "ro", mfc="none")
    if doPrint:
        plt.savefig(os.path.join(out_dir, "Fig1.png"), dpi=300)

    print("Ideal iteration number:")
    print("NCC:  " + str(max_NCCLoss))
    print(
        "Cross entropy:  "
        + str(max_CrossEntropyLoss)
        + "; relative error [%]:  "
        + str((max_CrossEntropyLoss - max_NCCLoss) / max_NCCLoss * 100)
    )
    print(
        "Poisson Loss:  "
        + str(max_PoissonLoss)
        + "; relative error [%]:  "
        + str((max_PoissonLoss - max_NCCLoss) / max_NCCLoss * 100)
    )

    # Save best deconvolution result for display
    deconv = est_history[max_NCCLoss, :, :]
    max_optim = max_PoissonLoss
    deconv_split = np.sum(est_split_history[max_optim], axis=0)

    plt.figure()
    ax1 = plt.subplot(231)
    plt.imshow(img, cmap="viridis")
    plt.axis("off")
    plt.title("Image", fontsize=10)
    plt.subplot(232, sharex=ax1, sharey=ax1)
    plt.imshow(img_T, cmap="viridis")
    plt.axis("off")
    plt.title("Split image 1", fontsize=10)
    plt.subplot(233, sharex=ax1, sharey=ax1)
    plt.imshow(img_V, cmap="viridis")
    plt.axis("off")
    plt.title("Split image 2", fontsize=10)
    plt.subplot(234, sharex=ax1, sharey=ax1)
    plt.imshow(obj, cmap="viridis")
    plt.axis("off")
    plt.title("Groundtruth", fontsize=10)
    plt.subplot(235, sharex=ax1, sharey=ax1)
    plt.imshow(deconv, cmap="viridis")
    plt.axis("off")
    plt.title("Knowing groundtruth", fontsize=10)
    plt.subplot(236, sharex=ax1, sharey=ax1)
    plt.imshow(deconv_split, cmap="viridis")
    plt.axis("off")
    plt.title("Binomial splitting", fontsize=10)
    plt.tight_layout()
    if doPrint:
        plt.savefig(os.path.join(out_dir, "Fig2.png"), dpi=300)

    if obj_name == "test_target":
        X = np.arange(0, np.shape(obj)[0]) * pxSize[0]
        plt.figure()
        ax = plt.subplot(313)
        plt.plot(X, obj[:, midPos[1]], color="0.5")
        plt.plot(X, deconv[:, midPos[1]], "r-")
        plt.xlim(0, np.shape(obj)[0] * pxSize[0])
        plt.xlabel("x-coordinate / (µm)", fontsize=10)
        plt.ylabel("Signal", fontsize=10)
        plt.title("Knowing groundtruth object", fontsize=8)
        plt.grid("on")
        plt.subplot(312, sharex=ax, sharey=ax)
        plt.plot(X, obj[:, midPos[1]], color="0.5")
        plt.plot(X, deconv_split[:, midPos[1]], "b-")
        plt.xlim(0, np.shape(obj)[0] * pxSize[0])
        plt.xlabel("x-coordinate / (µm)", fontsize=10)
        plt.ylabel("Signal", fontsize=10)
        plt.title("Cross-entropy criterion", fontsize=8)
        plt.grid("on")
        plt.subplot(311, sharex=ax, sharey=ax)
        plt.plot(X, obj[:, midPos[1]], color="0.5")
        plt.plot(X, deconv_split[:, midPos[1]], "k-")
        plt.xlim(0, np.shape(obj)[0] * pxSize[0])
        plt.xlabel("x-coordinate / (µm)", fontsize=10)
        plt.ylabel("Signal", fontsize=10)
        plt.title("Log. likelihood criterion", fontsize=8)
        plt.grid("on")
        plt.tight_layout()
        if doPrint:
            plt.savefig(os.path.join(out_dir, "Fig5a.png"), dpi=300)

        plt.figure()
        ax = plt.subplot(313)
        plt.plot(X, obj[:, int(midPos[1] / 4)], color="0.5")
        plt.plot(X, deconv[:, int(midPos[1] / 4)], "r-")
        plt.xlim(0, np.shape(obj)[0] * pxSize[0])
        plt.xlabel("x-coordinate / (µm)", fontsize=10)
        plt.ylabel("Signal", fontsize=10)
        plt.title("Knowing groundtruth object", fontsize=8)
        plt.grid("on")
        plt.subplot(312, sharex=ax, sharey=ax)
        plt.plot(X, obj[:, int(midPos[1] / 4)], color="0.5")
        plt.plot(X, deconv_split[:, int(midPos[1] / 4)], "b-")
        plt.xlim(0, np.shape(obj)[0] * pxSize[0])
        plt.xlabel("x-coordinate / (µm)", fontsize=10)
        plt.ylabel("Signal", fontsize=10)
        plt.title("Cross-entropy criterion", fontsize=8)
        plt.grid("on")
        plt.subplot(311, sharex=ax, sharey=ax)
        plt.plot(X, obj[:, int(midPos[1] / 4)], color="0.5")
        plt.plot(X, deconv_split[:, int(midPos[1] / 4)], "k-")
        plt.xlim(0, np.shape(obj)[0] * pxSize[0])
        plt.xlabel("x-coordinate / (µm)", fontsize=10)
        plt.ylabel("Signal", fontsize=10)
        plt.title("Log. likelihood criterion", fontsize=8)
        plt.grid("on")
        plt.tight_layout()
        if doPrint:
            plt.savefig(os.path.join(out_dir, "Fig5b.png"), dpi=300)

    # Compute Fourier error
    myMask = rr(numPixel, numPixel) < 2 * kAbbe
    FTest_history = ft2d(est_history)  # * myMask
    FTobj = ft2d(obj)  # * myMask

    FTError = abssqr(FTest_history - FTobj)

    N = np.arange(0, FTError.shape[0], 1)
    color = iter(plt.cm.rainbow(np.linspace(0, 1, N.shape[0])))
    cmap = "rainbow"

    k = np.arange(0, midPos[1]) / (2 * kAbbe)
    eps = 1e-12

    plt.figure()
    for l, c in zip(N, color):
        plt.semilogy(
            k,
            radialmean(FTError[l, :, :], center=(midPos[0], midPos[1]), nbins=midPos[0])
            + eps,
            color=c,
        )
    plt.semilogy(
        k,
        radialmean(
            FTError[max_NCCLoss, :, :], center=(midPos[0], midPos[1]), nbins=midPos[0]
        )
        + eps,
        "-",
        c="0.25",
        label="knowing groundtruth",
    )
    plt.semilogy(
        k,
        radialmean(
            FTError[max_optim, :, :], center=(midPos[0], midPos[1]), nbins=midPos[0]
        )
        + eps,
        ".",
        c="0.25",
        label="using binomial split",
    )

    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    plt.colorbar(sm, ticks=np.linspace(0, 1, 3), label="Relative iteration number")
    plt.grid("on")
    plt.xlabel("Spatial frequencies k / $k_{max}$")
    plt.ylabel("Spectral radial MSE")
    plt.legend(fontsize=8, loc="best")
    plt.xlim(0, 2)
    plt.ylim(1, None)
    plt.tight_layout
    if doPrint:
        plt.savefig(os.path.join(out_dir, "Fig3.png"), dpi=300)

    plt.figure()
    SumFTError = np.sum(FTError, axis=(1, 2))
    plt.plot(SumFTError, "g-")
    plt.xlabel("Iteration number")
    plt.ylabel("Integrated spectral MSE")
    plt.grid("on")
    ind = np.argmin(SumFTError)
    plt.plot(ind, SumFTError[ind], "go", label="minimal spectral MSE")
    plt.plot(
        max_NCCLoss,
        SumFTError[max_NCCLoss],
        "ro",
        mfc="none",
        label="knowing groundtruth",
    )
    plt.plot(
        max_PoissonLoss,
        SumFTError[max_PoissonLoss],
        "ko",
        mfc="none",
        label="likelihood criterion",
    )
    plt.plot(
        max_CrossEntropyLoss,
        SumFTError[max_CrossEntropyLoss],
        "bo",
        mfc="none",
        label="cross-entropy criterion",
    )
    plt.legend(fontsize=8, loc="best")
    plt.xlim(0, niter)
    plt.tight_layout()
    if doPrint:
        plt.savefig(os.path.join(out_dir, "Fig4.png"), dpi=300)
    # plt.gca().axes.yaxis.set_ticks([])

    plt.tight_layout()
    plt.show()

    if False:
        obj = np.ones_like(obj)
        img = np.random.poisson(obj * 10)
        img_T = np.random.binomial(img, coin_flip_bias)
        img_V = img - img_T
        plt.close("all")
        plt.hist(img.ravel(), bins="auto", label="Measurement")
        plt.hist(img_T.ravel(), bins="auto", label="Split image 1")
        plt.hist(img_V.ravel(), bins="auto", label="Split image 2")
        plt.xlabel("Measurement outcome")
        plt.ylabel("Number of occurrences")
        plt.grid("on")
        plt.tight_layout()
        plt.legend(fontsize=8, loc="best")
        plt.savefig(os.path.join(out_dir, "Fig0a.png"), dpi=300)

        plt.figure()
        for l in np.arange(9):
            plt.subplot(3, 3, l + 1)
            plt.scatter(np.random.permutation(img_T.ravel()), img_V.ravel(), marker=".")
            if l == 0:
                plt.xlabel("Split image 1", fontsize=9)
                plt.ylabel("Split image 2", fontsize=9)
            plt.title("Permutation " + str(l + 1), fontsize=8)
            # plt.gca().axes.xaxis.set_ticks([]);plt.gca().axes.yaxis.set_ticks([])
            plt.grid("on")
            plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "Fig0b.p"))
    # %%
