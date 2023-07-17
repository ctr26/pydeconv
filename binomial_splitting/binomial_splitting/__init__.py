from . import deconvolve
import numpy as np

def split(image,p=0.5):
    image_T = np.random.binomial(image, p)
    image_V = image - image_T
    return (image_T, image_V)