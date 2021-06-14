# pydeconv
Python package for scalable micrograph deconvolution with spatially varying point spread functions

<<<<<<< HEAD
## Build
  make build

## Install
  make install
=======
Disclaimer!
The presented idea is based on thoughts of Andrew York (@AndrewGYork; andrew.g.york+github@gmail.com), whos thoughts I then followed up.
>>>>>>> 2e1b897 (Update README.md)

When to stop an iterative image reconstruction? Especially when only a single noisy image has been recoreded.
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
=======
>>>>>>> dcf4af6 (Update README.md)
In general this depends on the signal-to-noise ratio (SNR) and on the underlying (unkown) object itself. What often happens with iterative image reconstruction is, that the
=======
For example: lets suppose we want to deconvolve a noisy image using the Richardson-Lucy algorithm ([I'm an inline-style link](https://www.google.com))
=======
For example: lets suppose we want to deconvolve a noisy image using the Richardson-Lucy algorithm ([Wikipedia:Richardson_Lucy](https://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution))
>>>>>>> 9dd244e (Update README.md)
=======

<<<<<<< HEAD
<<<<<<< HEAD
For example: lets suppose we want to deconvolve a shot-noise corrupted image using the Richardson-Lucy algorithm ([Wikipedia:Richardson_Lucy](https://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution)). The only parameter we really can adjust is the number of iterations. If we set it too low, we wont achieve much image reconstruction. When set to high the resulting reconstruction will exhibit amplified noise artifacts, simply because the iterative reconstruction starts to fit the noise as image structure.
>>>>>>> a258bcb (Update README.md)
In general this depends on the signal-to-noise ratio (SNR) and on the underlying (unknown) object itself.
>>>>>>> bb1da0c (Update README.md)
=======
For example: lets suppose we want to deconvolve a shot-noise corrupted image using the Richardson-Lucy algorithm ([Wikipedia:Richardson_Lucy](https://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution)). The only parameter we really can adjust is the number of iterations. If we set it too low, we wont achieve much image reconstruction. When set to high the resulting reconstruction will exhibit amplified noise artifacts, simply because the iterative reconstruction starts to fit the noise as image structure. 

In general the number of "optimal iteration" depends on the signal-to-noise ratio (SNR) and on the underlying (unknown) object itself. When only 
>>>>>>> 1db9cfa (Update README.md)
=======
For example: lets suppose we want to deconvolve a shot-noise corrupted image using the Richardson-Lucy algorithm ([Wikipedia:Richardson_Lucy](https://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution)). The only parameter we really need to adjust is the number of iterations. If we set it too low, we wont achieve much improvement in terms of reconstructing the underyling sample distribution. When set to high the resulting reconstruction will exhibit amplified noise artifacts, simply because the iterative reconstruction starts to fit the recorded noise as image structure. A compromise is often achieved by manually changing  the number of iterations and visually inspecting the results. 
In general the number of "optimal iteration" depends on the signal-to-noise ratio (SNR) and on the underlying (unknown) object itself. When the underlying object distribution is known (e.g. in a simulation) the normalized cross-correlation (NCC) is a good indicator of when to stop the iteration: after the NCC has reached its maximum the iterative image reconstruction will only fit noise. 
>>>>>>> 2e1b897 (Update README.md)

![alt text](https://github.com/beckerjn92/Binomial-splitting/blob/main/Fig1.PNG)
![alt text](https://github.com/beckerjn92/Binomial-splitting/blob/main/Fig2.PNG)
![alt text](https://github.com/beckerjn92/Binomial-splitting/blob/main/Fig3.PNG)
>>>>>>> 8f1dc3f (Update README.md)
=======
For example: lets suppose we want to deconvolve a shot-noise corrupted image using the Richardson-Lucy algorithm ([Wikipedia:Richardson_Lucy](https://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution)). The only parameter we really need to adjust is the number of iterations. If we set it too low, we wont achieve much improvement in terms of reconstructing the underyling sample distribution. When set to high the resulting reconstruction will exhibit amplified noise artifacts, simply because the iterative reconstruction starts to fit the recorded noise as image structure. A compromise is often achieved by manually changing  the number of iterations and visually inspecting the results. 
In general the number of "optimal iteration" depends on the signal-to-noise ratio (SNR) and on the underlying (unknown) object itself. When the underlying object distribution is known (e.g. in a simulation) the normalized cross-correlation (NCC) is a good indicator of when to stop the iteration: after the NCC has reached its maximum the iterative image reconstruction will only fit noise, hence as soon as the NCC decreases the algorithm should be stopped. However, this requires knowldedge of the underlying object, which reconstruction was the goal in the first place.  What to do when only a single image was recorded? Is there a way to estimate when to stop the iterative algorithm?

Yes there is!

![alt text](https://github.com/beckerjn92/Binomial-splitting/blob/main/Fig1.PNG)
![alt text](https://github.com/beckerjn92/Binomial-splitting/blob/main/Fig2.PNG)
>>>>>>> a4fbdfa (Update README.md)
