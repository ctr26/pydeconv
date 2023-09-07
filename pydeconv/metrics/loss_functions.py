import numpy as np
from . import Metric

# Nomenclature
# recorded_image
# gt
# estimate_history

EPS = 1e-12
# EPS = np.finfo(float).eps


# TODO check axes
def loglikelihood(est, gt, axis=(-2, -1)):
    return np.sum(
        -est + gt * np.log(est + EPS),
        axis=axis,
    )


def poisson(est, gt, axis=(-2, -1)):
    return np.sum(
        -est + gt * np.log(gt + EPS),
        axis=axis,
    )


#
def ncc(est, gt, axis=(-2, -1)):
    return np.squeeze(
        np.mean(
            (gt - np.mean(gt)) * (est, np.mean(est, axis=axis, keepdims=True)),
            axis=axis,
            keepdims=True,
        )
        / (np.std(gt) * np.std(est, axis=axis, keepdims=True))
    )


def crossentropy(est, gt, axis=(-2, -1)):
    return np.sum(
        est * np.log(gt + EPS),
        axis=axis,
    )


def kl(est, gt, axis=(-2, -1)):
    return kl_div(p=gt, q=est, axis=axis)
    # return np.sum(
    # fwd(obj) * np.log((fwd(obj) + EPS) / (fwd(estimate) + EPS)),
    # axis=(1, 2),
    # )


def kl_div(p, q, axis=(-2, -1)):
    return np.sum(p * (np.log((p + EPS) / (q + EPS))), axis=axis)


# CrossEntropy and PoissonLoss of not ground truth

# kl_est_noiseless_signal = np.sum(
#     np.expand_dims(fwd(obj), 0)
#     * np.log((np.expand_dims(fwd(obj), 0) + 1e-9) / (fwd(est_history) + 1e-9)),
#     axis=(1, 2),
# )

# The following should be the ground truth best kl divergence of

# kl_est_noiseless_signal = np.sum(
#     fwd(obj) * np.log((fwd(obj) + 1e-9) / (fwd(est_history) + 1e-9)),
#     axis=(1, 2),
# )
