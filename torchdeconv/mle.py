import torch

def p_x_given_b(b: torch.tensor, Ax: torch.tensor):
    # This is log(p(x|b))
    # This is true for poisson only noise
    # b = b + 1e-6
    Ax = Ax + 1e-6
    log_b_factorial = torch.lgamma(b + 1)
    return torch.multiply(torch.log(Ax), (b)) - Ax - log_b_factorial

def log_liklihood_x_given_b(b, Ax):
    return -torch.sum(p_x_given_b(b, Ax))


def kl_divergence(p, q):
    # predictions = predictions + 1e-6 / H_T_1
    # p = prediction
    # q = observation
    p = p + 1e-6 / p.sum()
    # true = true + 1e-6 / H_T_1
    q = q + 1e-6 / q.sum()
    # predictions = torch.nn.functional.log_softmax(predictions, dim=0)
    # true = torch.nn.functional.softmax(true,dim=0)
    kl_div = (p * ((p / q).log())).sum()
    # kl_div = torch.nn.functional.kl_div(predictions.log(), true, reduction="sum")
    return kl_div


def js_divergence_torch(p, q):
    return (
        kl_divergence(p, q) + kl_divergence(q, p)
    ) / 2
