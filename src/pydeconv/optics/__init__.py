from . import utils
import numpy as np


class OpticalModel:
    def __init__(self, psf):
        self.psf = psf
        self.otf = psf2otf(psf)

    def fwd(self, x):
        return forward_optical_model(x, self.otf)

    # return np.real(utils.ift2d(utils.ft2d(x) * self.otf))
    def bwd(self, x):
        return backward_optical_model(x, self.otf)
        # return np.real(utils.ift2d(utils.ft2d(x) * np.conj(self.otf)))


# apsf = jinc_psf(numPixel, midPos, pxSize, lambda0, na)


def operators(
    psf,
    numPixel=128,
    midPos=64,
    pxSize=0.1,
    lambda0=0.5,
    na=0.5,
):
    # TODO nd
    # psf = jincPSF(numPixel, midPos, pxSize, lambda0, NA)
    # obj = obj / np.max(obj) * max_photons
    # Forward and backwards model

    otf = psf2otf(psf)
    optical_model = OpticalModel(psf)
    # fwd = lambda x: np.real(utils.ift2d(utils.ft2d(x) * otf))
    # bwd = lambda x: np.real(utils.ift2d(utils.ft2d(x) * np.conj(otf)))
    return otf, optical_model.fwd, optical_model.bwd


def psf2otf(apsf):
    psf = apsf2psf(apsf)
    otf = utils.ft2d(psf)
    return otf


def apsf2psf(apsf):
    psf = utils.abssqr(apsf)
    psf = psf / np.sum(psf) * np.sqrt(np.size(psf))
    return psf


def forward_optical_model(x, otf):
    fwd = np.real(utils.ift2d(utils.ft2d(x) * otf))
    return np.clip(fwd, 0, np.inf)


def backward_optical_model(x, otf):
    # TODO UNSURE IF THIS CLIPS
    return np.real(utils.ift2d(utils.ft2d(x) * np.conj(otf)))


def abbe_radius_from_pupil(pupil, midPos):
    arr = np.argwhere(utils.abssqr(pupil / np.max(pupil)) > 1e-3)
    R = np.max(np.sqrt((arr[:, 0] - midPos[0]) ** 2 + (arr[:, 1] - midPos[1]) ** 2))
    return np.ceil(R)


def simulate_image(obj, fwd, noise=True):
    img = fwd(obj)
    img = np.clip(img, 0, None)
    # Apply shot noise
    if noise:
        return np.random.poisson(img).astype("int32")
    return img


def simulate_image_SNR(data, fwd, snr, seed=42):
    # def corrupt_data(data: NDArray[Int], psf: NDArray[Float], gain, snr):
    # data_blur = conv2(data, psf*gain, "same")  # Blur image
    # TODO correct fwd for gain
    data_blur = fwd(data)
    rng = np.random.default_rng()

    target_snr_db = snr
    # Calculate signal power and convert to dB
    sig_avg_watts = np.mean(data_blur)
    sig_avg_db = 10 * np.log10(sig_avg_watts)

    noise_avg_db = sig_avg_db - target_snr_db
    noise_avg_photons = 10 ** (noise_avg_db / 10)

    mean_noise = 0
    white_noise = np.random.normal(
        mean_noise, np.sqrt(noise_avg_photons), data_blur.shape
    )

    signal = (rng.poisson(data_blur) + white_noise).clip(0).astype(int)
    return signal
