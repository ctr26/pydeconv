from . import richardson_lucy

def deconvolve(image,psf,method="richardson_lucy"):
    if method == "richardson_lucy":
        return richardson_lucy.deconvolve(image,psf)
    

def weiner(image,psf):
    pass