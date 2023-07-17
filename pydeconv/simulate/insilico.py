from scipy import ndimage
import numpy as np
from skimage import transform

# import rescale, resize, downscale_local_mean
from skimage import color, data, restoration, exposure
from skimage.metrics import peak_signal_noise_ratio, structural_similarity as ssim
from pydeconv import optics, utils


class CorruptImage:
    psf_w, psf_h, sigma, scale = 64, 64, 1, 4  # Constants
    stop_loss = 1e-2
    step_size = 20 * stop_loss / 3.0

    gain = (2**16) / 10

    def __init__(
        self,
        data,
        snr=np.inf,
        psf={"kind": "gaussian", "width": 64, "height": 64},
        seed=42,
    ):
        self.gt = get_image(data)
        self.psf = generate_psf(psf)
        self.signal = optics.simulate_image_SNR(
            data=self.gt, fwd=self.psf, snr=snr, seed=seed
        )

    def get_data(self):
        return self.gt, self.signal, self.psf


def generate_psf(psf_dict):
    kind = psf_dict["kind"]
    psf_w = psf_dict["width"]
    psf_h = psf_dict["height"]

    if kind == "gaussian":
        return get_gaussian_psf(psf_w=psf_w, psf_h=psf_h)
    return None


def get_image(arr=data.astronaut(), scale=1):
    image = color.rgb2gray(arr)
    return transform.rescale(image, 1.0 / scale)


def get_gaussian_psf(psf_w=64, psf_h=64, sigma=1):
    psf = np.zeros((psf_w, psf_h))
    psf[psf_w // 2, psf_h // 2] = 1
    # psf = gaussian_filter(psf, sigma=sigma)  # PSF
    return ndimage.gaussian_filter(psf, sigma=sigma)
