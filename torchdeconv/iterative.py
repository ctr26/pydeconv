from functools import cache
from torch import nn,optim
from tqdm import tqdm

# class Aesthics():
#     def __init__(aesthics):
class Iterator():
    steps = 100
    def __init__(self,img,psf,steps=100):
        self.img = img
        self.psf = psf
        # self.method = method
        # self.optimiser = optimiser
        # self.loss = loss
        self.steps = steps
        
    @cache
    def iterate(self,n,pbar):
        pbar.update(n)
        if n == 0:
            return self.img
        return self.step(self.iterate(n-1,pbar))
    
    @cache
    def step(self,img):
        pass

    @cache
    def gradient(self,n):
        pass
    
    @cache
    def aesthics(self,n):
        pass
    
    def early_stopping(self):
        pass
    
    def solve(self):
        # pbar = tqdm(range(0,self.steps))
        pbar = tqdm(total=self.steps)
        return self.iterate(self.steps,pbar)

        

