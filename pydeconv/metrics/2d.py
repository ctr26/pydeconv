
import numpy as np

# Nomenclature
# recorded_image
# gt
# estimate_history

EPS = np.finfo(float).eps


# TODO check axes
def loglikelihood(estimate, image, fwd):
    return np.sum(
        -fwd(estimate) + image * np.log(fwd(estimate) + EPS),
        axis=(1, 2, 3),
    )


def poissonloss(estimate, image, fwd):
    return np.sum(
        -fwd(estimate) + image * np.log(fwd(estimate) + EPS),
        axis=(1, 2, 3),
    )


#
def nccloss(estimate, gt):
    return np.squeeze(
        np.mean(
            (gt - np.mean(gt))
            * (estimate, np.mean(estimate, axis=(1, 2), keepdims=True)),
            axis=(1, 2),
            keepdims=True,
        )
        / (np.std(gt) * np.std(estimate, axis=(1, 2), keepdims=True))
    )


def crossentropyloss(estimate, image, fwd):
    return np.sum(
        estimate * np.log(estimate + EPS),
        axis=(1, 2, 3),
    )

def kl_noiselss_signal(estimate, obj, image, fwd):
    return np.sum(
        fwd(obj) * np.log((fwd(obj) + EPS) / (fwd(estimate) + EPS)),
        axis=(1, 2),
    )


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