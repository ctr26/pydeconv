import numpy as np
from torchdeconv import Deconvolver
import PIL
import pytest

img = np.ones((128,128))
psf = np.ones((5,5))

@pytest.mark.parametrize("img", [PIL.Image.fromarray(img)])
@pytest.mark.parametrize("psf", [PIL.Image.fromarray(psf)])
def test_deconvolve(img,psf):
    img = Deconvolver(img)
    decon_img = img.deconvolve(psf)
    assert decon_img.size == img.size
    