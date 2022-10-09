import torch
from torch import nn
from . import mle
class HTorch(nn.Module):
    def __init__(self, x_0,H,device="cpu"):
        super(HTorch, self).__init__()
        self.H = H
        self.x_0 = x_0
        # self.x = torch.nn.Linear(H.shape[0], 1)
        self.x_torch = torch.tensor(
            x_0, requires_grad=True, dtype=torch.float, device=device
        )  # Requires grad is magic
        self.x = torch.nn.Parameter(self.x_torch)
        self.y_pred = self.get_y_pred()

    # def __call__(self,x):
    #     return torch.matmul(H_torch, x).double()
    def forward(self):
        # self.x = torch.nn.Parameter((self.x>0)*self.x)
        # self.img_T = torch.tensor(np.random.binomial(self.y_pred.int().detach().numpy(),0.5)).float().requires_grad_()
        self.y_pred = self.get_y_pred()
        # self.img_T = torch.distributions.binomial.Binomial(y_pred.int(),0.5*torch.ones_like(y_pred)).sample()
        # self.img_T = torch.tensor(np.random.binomial(y.double(),0.5)).double().requires_grad_()
        # # self.img_T = torch.tensor(np.random.binomial(y.double(),0.5)).double()
        return self.predict(self.x)

    def predict(self, x):
        # self.x = torch.nn.Parameter((self.x>0)*self.x)
        # return
        return torch.matmul(self.H.double(), x.double()).double()

    def get_VT(self):
        img_T = torch.distributions.binomial.Binomial(
            self.y_pred.int(), 0.5 * torch.ones_like(self.y_pred)
        ).sample()
        img_V = self.y_pred - img_T
        return (img_T, img_V)

    def get_y_pred(self):
        return self.predict(self.x).float()
    
