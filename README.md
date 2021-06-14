# Binomial-splitting
Employ binomial splitting to create two stochastically independent images from a single measure image.

Disclaimer!
The presented idea is based on thoughts of Andrew York (@AndrewGYork; andrew.g.york+github@gmail.com) who gratefully shared this with me.

When to stop an iterative image reconstruction? Especially when only a single noisy image has been recoreded.
For example: lets suppose we want to deconvolve a noisy image using the Richardson-Lucy algorithm ([I'm an inline-style link](https://www.google.com))
In general this depends on the signal-to-noise ratio (SNR) and on the underlying (unknown) object itself.

![alt text](https://github.com/beckerjn92/Binomial-splitting/blob/main/Fig1.PNG)
![alt text](https://github.com/beckerjn92/Binomial-splitting/blob/main/Fig2.PNG)
![alt text](https://github.com/beckerjn92/Binomial-splitting/blob/main/Fig3.PNG)
