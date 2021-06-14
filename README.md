# pydeconv
Python package for scalable micrograph deconvolution with spatially varying point spread functions

## Build
  make build

## Install
  make install

When to stop an iterative image reconstruction? Especially when only a single noisy image has been recoreded.
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
In general this depends on the signal-to-noise ratio (SNR) and on the underlying (unkown) object itself. What often happens with iterative image reconstruction is, that the
=======
For example: lets suppose we want to deconvolve a noisy image using the Richardson-Lucy algorithm ([I'm an inline-style link](https://www.google.com))
=======
For example: lets suppose we want to deconvolve a noisy image using the Richardson-Lucy algorithm ([Wikipedia:Richardson_Lucy](https://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution))
>>>>>>> 9dd244e (Update README.md)
=======

<<<<<<< HEAD
For example: lets suppose we want to deconvolve a shot-noise corrupted image using the Richardson-Lucy algorithm ([Wikipedia:Richardson_Lucy](https://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution)). The only parameter we really can adjust is the number of iterations. If we set it too low, we wont achieve much image reconstruction. When set to high the resulting reconstruction will exhibit amplified noise artifacts, simply because the iterative reconstruction starts to fit the noise as image structure.
>>>>>>> a258bcb (Update README.md)
In general this depends on the signal-to-noise ratio (SNR) and on the underlying (unknown) object itself.
>>>>>>> bb1da0c (Update README.md)
=======
For example: lets suppose we want to deconvolve a shot-noise corrupted image using the Richardson-Lucy algorithm ([Wikipedia:Richardson_Lucy](https://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution)). The only parameter we really can adjust is the number of iterations. If we set it too low, we wont achieve much image reconstruction. When set to high the resulting reconstruction will exhibit amplified noise artifacts, simply because the iterative reconstruction starts to fit the noise as image structure. 

In general the number of "optimal iteration" depends on the signal-to-noise ratio (SNR) and on the underlying (unknown) object itself. When only 
>>>>>>> 1db9cfa (Update README.md)

![alt text](https://github.com/beckerjn92/Binomial-splitting/blob/main/Fig1.PNG)
![alt text](https://github.com/beckerjn92/Binomial-splitting/blob/main/Fig2.PNG)
![alt text](https://github.com/beckerjn92/Binomial-splitting/blob/main/Fig3.PNG)
>>>>>>> 8f1dc3f (Update README.md)
