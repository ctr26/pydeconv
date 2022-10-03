import numpy as np
from torchdeconv import Deconvolver
import PIL
import pytest

img = np.ones((128,128))
psf = np.ones((5,5))

@pytest.mark.parametrize("img", [img,PIL.Image.fromarray(img)])
@pytest.mark.parametrize("psf", [psf,PIL.Image.fromarray(psf)])
def test_deconvolve(img,psf):
    img = Deconvolver(img)
    img.deconvolve(psf)
