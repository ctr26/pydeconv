<<<<<<< HEAD
<<<<<<< HEAD
# pydeconv
Python package for scalable micrograph deconvolution with spatially varying point spread functions
=======
# Binomial-splitting: estimating the optimal iteration number in image reconstruction
Employ binomial splitting to create two stochastically independent images from a single measure image.
>>>>>>> 59cfad1 (Update README.md)
=======
# Binomial-splitting: When to stop iterative image reconstruction?
>>>>>>> 8b4e491 (Update README.md)

<<<<<<< HEAD
## Build
  make build

## Install
  make install
=======
Disclaimer!
<<<<<<< HEAD
<<<<<<< HEAD
The presented idea is based on thoughts of Andrew York (@AndrewGYork; andrew.g.york+github@gmail.com), whos thoughts I then followed up.
>>>>>>> 2e1b897 (Update README.md)
=======
The presented idea is based on thoughts of Andrew York (@AndrewGYork; andrew.g.york+github@gmail.com), whos thoughts I then followed up. The presented code is written in MATLAB using the DipImage toolbox ([here](https://diplib.org/)). A Python version is to be added soon.
>>>>>>> 7a3f90f (Update README.md)
=======
The presented idea is based on thoughts of Andrew York (@AndrewGYork; andrew.g.york+github@gmail.com), who's thoughts I then followed up and discussed with Craig Russel (@microRussel; ctr26@ebi.ac.uk). The presented code is written in MATLAB using the DipImage toolbox ([here](https://diplib.org/)). A Python version is to be added soon (.
>>>>>>> 62fb3c9 (Update README.md)

When to stop an iterative image reconstruction? Especially when only a single noisy image has been recoreded.
<<<<<<< HEAD
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
=======
=======
>>>>>>> f40342503aceedf8cfdc3f15de7ecf667b43c9c2
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

Yes there is! The general idea is borrowed from "machine learning". We want to split the recorded data, which is corrupted by shot noise, into a training and validation batch. How is this done? By simply applying a binomial distribution on top of the Poisson-distributed image data (see the pdf for a detailed explanation). With this we are able to obtain two stochastically independent images from a single recorded image.  
![alt text](https://github.com/beckerjn92/Binomial-splitting/blob/main/Fig1.PNG)
We can use this to estimate when to stop the iterative image reconstruction. In a first step we compute the ordinary RL-deconvolution with the "training"-batch. For each iteration we compute the negative log-likelihood (- ln[L]) and observe that it is a monotonically increasing function. Just as we expect, since the RL-algorithm is supposed to maximize -ln[L]. What happens now if we compute -ln[L], but instead of comparing to the "training" we refer now to the "validation" data? Because "training" and "validation" are stochastically independend, we suddenly see a similar curve as for the NCC-curve. Interestingly, both maxima roughly coincide meaning that it might be possible to estimate the optimum iteration number without the requirement of knowing the unknown object.
![alt text](https://github.com/beckerjn92/Binomial-splitting/blob/main/Fig2.PNG)
>>>>>>> a4fbdfa (Update README.md)
