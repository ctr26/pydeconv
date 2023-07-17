from typing import Any
import pandas as pd
from pydeconv.simulate import psfs
from pydeconv import optics, utils
from types import SimpleNamespace

# class ReconstructionMetrics:
# def __init__(self):
# self.estimate = estimate
# self.image = estimate[0]
# self.otf = optics.psf2otf(psf)
from tqdm import tqdm


def get_metric(metric):
    return metric()


def get_metrics(metrics=None):
    return SimpleNamespace(**get_metrics_dict(metrics))


def get_metrics_dict(metrics=None):
    metrics_dict = {}
    for metric in tqdm(metrics):
        name = type(metric).__name__
        metrics_dict[name] = get_metric(metric)
    return metrics_dict


def get_metrics_df(metrics=None):
    return pd.DataFrame(get_metrics_dict(metrics))

    # TODO something like how scikit-learn does it


import numpy as np
from dataclasses import dataclass


@dataclass
class Metric:
    x_gt: np.ndarray
    y_gt: np.ndarray
    x_est: np.ndarray
    y_est: np.ndarray
    axis: tuple
    # fwd: Any = None
    metric_fn: Any = None

    def __call__(
        self,
    ) -> Any:
        return self.metric()

    def metric(self) -> Any:
        return self.metric_fn(
            self.x_gt, self.y_gt, self.x_est, self.y_est, axis=self.axis
        )


from . import loss_functions


class ReconstructionMetrics2D(Metric):
    axis = (-2, -1)

    def __init__(self, axis=(-2, -1), **kwargs):
        self.loglikelihood = Loglikelihood(axis=axis, **kwargs)
        self.poisson = Poisson(axis=axis, **kwargs)
        self.ncc = NCC(axis=axis, **kwargs)
        self.crossentropy = CrossEntropy(axis=axis, **kwargs)
        self.kl_noiseless_signal = KLNoiselessSignal(axis=axis, **kwargs)
        self.all = [
            self.loglikelihood,
            self.poisson,
            self.ncc,
            self.crossentropy,
            self.kl_noiseless_signal,
        ]


class Loglikelihood(Metric):
    def metric(self):
        return loss_functions.loglikelihood(
            y_est=self.y_est,
            y_gt=self.y_gt,
            axis=self.axis,
        )


class Poisson(Metric):
    def metric(self):
        return loss_functions.poisson(
            y_est=self.y_est,
            y_gt=self.y_gt,
            axis=self.axis,
        )


class NCC(Metric):
    def metric(self):
        return loss_functions.ncc(
            x_est=self.x_est,
            x_gt=self.x_gt,
            axis=self.axis,
        )


class CrossEntropy(Metric):
    def metric(self):
        return loss_functions.crossentropy(
            y_est=self.y_est,
            y_gt=self.y_gt,
            axis=self.axis,
        )


class KLNoiselessSignal(Metric):
    def metric(self):
        return loss_functions.kl_noiseless_signal(
            y_est=self.y_est,
            y_gt=self.y_gt,
            axis=self.axis,
        )
