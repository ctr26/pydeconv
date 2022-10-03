import numpy as np
from torchdeconv import Deconvolver
import PIL
img = np.ones((128,128))
# img = PIL.Image.fromarray(np.ones((128,128)))

def test_deconvolve(img,psf):
    img = Deconvolver(img)
    img.deconvolve(psf)
