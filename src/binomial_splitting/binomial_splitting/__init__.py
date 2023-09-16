from . import deconvolve
import numpy as np

def split(image,p=0.5,norm=False):
    image_T = np.random.binomial(image, p)
    image_V = image - image_T
    if norm:
        return (image_T/p, image_V/p)
    return (image_T, image_V)