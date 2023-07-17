import numpy as np
from . import Metric

# Nomenclature
# recorded_image
# gt
# estimate_history

EPS = 1e-12
# EPS = np.finfo(float).eps


# TODO check axes
def loglikelihood(y_est, y_gt, axis=(-2, -1)):
    return np.sum(
        -y_est + y_gt * np.log(y_est + EPS),
        axis=axis,
    )


def poisson(y_est, y_gt, axis=(-2, -1)):
    return np.sum(
        -y_est + y_gt * np.log(y_gt + EPS),
        axis=axis,
    )


#
def ncc(x_est, x_gt, axis=(-2, -1)):
    return np.squeeze(
        np.mean(
            (x_gt - np.mean(x_gt)) * (x_est, np.mean(x_est, axis=axis, keepdims=True)),
            axis=axis,
            keepdims=True,
        )
        / (np.std(x_gt) * np.std(x_est, axis=axis, keepdims=True))
    )


def crossentropy(y_est, y_gt, axis=(-2, -1)):
    return np.sum(
        y_est * np.log(y_gt + EPS),
        axis=axis,
    )


def kl_noiseless_signal(y_est, y_gt, axis=(-2, -1)):
    return kl_div(p=y_gt, q=y_est, axis=axis)
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
