import torch.nn as nn
import os
import utils
import scipy
from scipy.ndimage import gaussian_filter
import numpy as np
from skimage.transform import rescale, resize, downscale_local_mean
from skimage import color, data, restoration, exposure
from scipy.linalg import circulant
from scipy.sparse import diags
from scipy.signal import convolve2d as conv2
import torch
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import utils
import pyro
from torch.distributions import constraints
import pyro.distributions as dist
import torch.distributions.constraints as constraints
import math
import os
import torch
import torch.distributions.constraints as constraints
from pyro.optim import Adam, SGD
from pyro.infer import SVI, Trace_ELBO
import pyro.distributions as dist
from scipy.linalg import circulant
from pyro.ops.tensor_utils import convolve
import matplotlib
import pandas as pd
import xarray as xr
from sklearn.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio, structural_similarity as ssim
from nptyping import NDArray, Bool, Int, Float



psf_w, psf_h, sigma, scale = 64, 64, 1, 4  # Constants
stop_loss = 1e-2
step_size = 20 * stop_loss / 3.0

# gain = (2**16) / 10


class CorruptImage:
    def __init__(
        self,
        data,
        gain=1000,
        snr=np.inf,
        psf={"kind": "gaussian", "width": 64, "height": 64},
        seed=42,
        ):
        self.gt = get_image(data)
        self.psf = generate_psf(psf)
        self.signal = corrupt_data(data=self.gt,psf=self.psf,gain=gain,snr=snr,seed=seed)
    
    def get_data(self):
        return self.gt,self.signal,self.psf
        
def generate_psf(psf_dict):
    kind = psf_dict["kind"]
    psf_w=psf_dict["width"]
    psf_h=psf_dict["height"]
    
    if kind == "gaussian":
        return get_gaussian_psf(psf_w=psf_w,psf_h=psf_h)
    return None
    

def get_image(self, arr=data.astronaut()):
    image = color.rgb2gray(arr)
    return rescale(image, 1.0 / scale)


def get_gaussian_psf(psf_w=64, psf_h=64, sigma=1):
    psf = np.zeros((psf_w, psf_h))
    psf[psf_w // 2, psf_h // 2] = 1
    # psf = gaussian_filter(psf, sigma=sigma)  # PSF
    return gaussian_filter(psf, sigma=sigma)


def corrupt_data(data, psf, gain, snr,seed=42):
# def corrupt_data(data: NDArray[Int], psf: NDArray[Float], gain, snr):
    data_blur = conv2(data, psf*gain, "same")  # Blur image
    rng = np.random.default_rng()

    target_snr_db = snr
    # Calculate signal power and convert to dB
    sig_avg_watts = np.mean(data_blur)
    sig_avg_db = 10 * np.log10(sig_avg_watts)

    noise_avg_db = sig_avg_db - target_snr_db
    noise_avg_photons = 10 ** (noise_avg_db / 10)

    mean_noise = 0
    white_noise = np.random.normal(mean_noise, np.sqrt(noise_avg_photons), data_blur.shape)

    signal = (rng.poisson(data_blur) + white_noise).clip(0).astype(int)
    return signal
