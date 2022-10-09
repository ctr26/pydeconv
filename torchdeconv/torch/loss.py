import torch
from torch import nn
from torchdeconv import mle

class NLL_Poisson(nn.Module):
    def __init__(self):
        super(NLL_Poisson, self).__init__()

    def forward(self, predictions, true,H_T_1):
        return mle.log_liklihood_x_given_b(true / H_T_1, predictions / H_T_1)
        # return log_liklihood_x_given_b(true, predictions)
        # return log_liklihood_x_given_b(predictions/H_T_1, true/H_T_1)
        # return log_liklihood_x_given_b(predictions,true)


class KLD(nn.Module):
    def __init__(self):
        super(KLD, self).__init__()

    def forward(self, predictions, true,H_T_1):
        return mle.kl_divergence_torch(predictions / H_T_1, true / H_T_1)


class JSD(nn.Module):
    def __init__(self):
        super(JSD, self).__init__()

    def forward(self, predictions, true,H_T_1):
        return mle.js_divergence_torch(predictions / H_T_1, true / H_T_1)


# class NL_KL(nn.Module):
#     def __init__(self):
#         super(NL_KL, self).__init__()

#     def forward(self, predictions, true):
#         return -(kl_divergence_torch(predictions, true).log())


class NMSE(nn.Module):
    def __init__(self):
        super(NMSE, self).__init__()

    def forward(self, predictions, true,H_T_1):
        return torch.sqrt(torch.nn.functional.mse_loss(predictions + 1e-6, true + 1e-6))


class L1_and_L2(nn.Module):
    def __init__(self):
        super(L1_and_L2, self).__init__()

    def forward(self, predictions, true,H_T_1):
        return torch.sqrt(
            torch.nn.functional.mse_loss(predictions + 1e-6, true + 1e-6)
        ) + torch.nn.functional.l1_loss(predictions + 1e-6, true + 1e-6)


class L1_and_L2_and_MLE(nn.Module):
    def __init__(self):
        super(L1_and_L2_and_MLE, self).__init__()

    def forward(self, predictions, true, H_T_1):
        return (
            torch.sqrt(torch.nn.functional.mse_loss(predictions + 1e-6, true + 1e-6))
            + torch.nn.functional.l1_loss(predictions + 1e-6, true + 1e-6)
            + mle.log_liklihood_x_given_b(true / H_T_1, predictions / H_T_1).log()
        )


class MLE_and_PoissonSplit(nn.Module):
    def __init__(self):
        super(MLE_and_PoissonSplit, self).__init__()

    def forward(self, predictions, true,img_V):
        return NLL_Poisson()(predictions, true) + NLL_Poisson()(predictions, img_V)


class L1(nn.Module):
    def __init__(self):
        super(L1, self).__init__()

    def forward(self, predictions, true):
        return torch.nn.functional.l1_loss(predictions, true)

