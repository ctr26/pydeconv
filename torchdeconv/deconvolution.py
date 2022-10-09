from . import richardson_lucy
from . import weiner
import PIL
import torch
import numpy as np
import torchvision

METHODS = {
            "richardson_lucy":richardson_lucy.deconvolve,
            "weiner":weiner.deconvolve,
           }

def deconvolve(image:PIL.Image.Image,psf:PIL.Image.Image,method="richardson_lucy"):
    # image_torch = torch.tensor(np.array(image))
    image_torch = torchvision.transforms.functional.pil_to_tensor(image)
    psf_torch = torchvision.transforms.functional.pil_to_tensor(psf)
    return METHODS[method](image_torch,psf_torch)

# def weiner(image,psf):
#     pass