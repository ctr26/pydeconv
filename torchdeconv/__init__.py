import PIL
from PIL import Image

from . import deconvolution

class Deconvolver():
    def __new__(cls,image,method="richardson_lucy"):
        return Image.fromarray(image)
    
    def __init__(self,image,method="richardson_lucy",*args,**kwargs):
        pass
    
    def deconvolve(self,psf,method="richardson_lucy",*args,**kwargs):
        return deconvolution.deconvolve(self,psf,*args,**kwargs)

    
        
# https://stackoverflow.com/questions/3209233/how-to-replace-an-instance-in-init-with-a-different-object

# class ClassA(object):
# def __new__(cls,theirnumber):
#     if theirnumber > 10:
#         # all big numbers should be ClassB objects:
#         return ClassB.ClassB(theirnumber)
#     else:
#         # numbers under 10 are ok in ClassA.
#         return super(ClassA, cls).__new__(cls, theirnumber)