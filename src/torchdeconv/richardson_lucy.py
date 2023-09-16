from tqdm import tqdm
import numpy as np
import torch 
from torchdeconv.iterative import Iterator

def deconvolve(img: torch.tensor,psf:torch.tensor):
    return RichardLucyIterator(img,psf).solve()
    # for i in pbar:
    #     img = richardson_lucy_step(img, psf)


def richardson_lucy_gradient(img: torch.tensor,psf:torch.tensor):
    return img

def richardson_lucy_step(img: torch.tensor,psf:torch.tensor,gradient_fun):
    return img*gradient_fun(img,psf)


def richardson_lucy_gradient_matrix(img: torch.tensor,H:torch.tensor,device="cpu"):
    H_T_1 = torch.matmul(
    torch.t(H),
    torch.ones_like(torch.tensor(img.flatten(), device=device)).float(),
)
    rl_step = np.matmul(H.transpose(), img / np.matmul(H, img)) / H_T_1.cpu()
    return rl_step


def richardson_lucy_learning_rate(rl_step):
    # TODO rethink this
    """Factor of 2 probably comes from nyquist sampling the gradient space

    Args:
        rl_step (_type_): _description_

    Returns:
        _type_: _description_
    """
    return rl_step.sum() / (2 * len(rl_step))



class RichardLucyIterator(Iterator):
    def __init__(self,img,psf,matrix_mode=False):
        self.img = img
        self.psf = psf
        self.matrix_mode = matrix_mode
        self.richardson_lucy_step = staticmethod(richardson_lucy_step)

    def step(self,img):
        if not self.matrix_mode:
            return richardson_lucy_step(img,self.psf,richardson_lucy_gradient)
        return richardson_lucy_step(img,self.psf,richardson_lucy_gradient_matrix)
    
    def gradient(self,n):
        img = self.iterate(n)
        if not self.matrix_mode:
            return richardson_lucy_gradient(img,self.psf)
        return richardson_lucy_gradient_matrix(img,self.psf)
    
    