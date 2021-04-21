import pytest

from skimage import data
from skimage import color



# get astronaut from skimage.data in grayscale
l = color.rgb2gray(data.astronaut())

data = 