# Binomial-splitting
Employ binomial splitting to create two stochastically independent images from a single measure image.

Disclaimer!
The presented idea is based on thoughts of Andrew York (@AndrewGYork; andrew.g.york+github@gmail.com) who gratefully shared this with me.

When to stop an iterative image reconstruction? Especially when only a single noisy image has been recoreded.

For example: lets suppose we want to deconvolve a shot-noise corrupted image using the Richardson-Lucy algorithm ([Wikipedia:Richardson_Lucy](https://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution)). The only parameter we really can adjust is the number of iterations. If we set it too low, we wont achieve much image reconstruction. When set to high the resulting reconstruction will exhibit amplified noise artifacts, simply because the iterative reconstruction starts to fit the noise as image structure.
In general this depends on the signal-to-noise ratio (SNR) and on the underlying (unknown) object itself.

![alt text](https://github.com/beckerjn92/Binomial-splitting/blob/main/Fig1.PNG)
![alt text](https://github.com/beckerjn92/Binomial-splitting/blob/main/Fig2.PNG)
![alt text](https://github.com/beckerjn92/Binomial-splitting/blob/main/Fig3.PNG)
