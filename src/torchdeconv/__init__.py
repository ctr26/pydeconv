import PIL
from PIL import Image
from functools import wraps # This convenience func preserves name and docstring
import pims

from . import deconvolution

import torchvision
# def add_method(cls):
#     def decorator(func):
#         @wraps(func) 
#         def wrapper(self, *args, **kwargs): 
#             return func(*args, **kwargs)
#         setattr(cls, func.__name__, wrapper)
#         # Note we are not binding func, but wrapper which accepts self but does exactly the same as func
#         return func # returning func means func can still be used normally
#     return decorator


class Deconvolver(PIL.Image.Image):
    # def __new__(cls,image):
        # obj = Image.fromarray(image)
        # obj.__new__(cls)
        # return super(Deconvolver).__new__(obj.__class__)
    def __init__(self,image):
        super().__init__()
        self.__dict__ = image.__dict__.copy()
    def deconvolve(self,psf,method="richardson_lucy",*args,**kwargs):
        decon_torch = deconvolution.deconvolve(self,psf,*args,**kwargs)
        return torchvision.transforms.functional.to_pil_image(decon_torch)

# class Deconvolver(pims):

        
# https://stackoverflow.com/questions/3209233/how-to-replace-an-instance-in-init-with-a-different-object

# class ClassA(object):
# def __new__(cls,theirnumber):
#     if theirnumber > 10:
#         # all big numbers should be ClassB objects:
#         return ClassB.ClassB(theirnumber)
#     else:
#         # numbers under 10 are ok in ClassA.
#         return super(ClassA, cls).__new__(cls, theirnumber)