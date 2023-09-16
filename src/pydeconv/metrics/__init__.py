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
        label = metric.label
        key = f"{name}{label}"
        metrics_dict[key] = get_metric(metric)
    return metrics_dict


def get_metrics_df(metrics=None):
    return pd.DataFrame(get_metrics_dict(metrics))

    # TODO something like how scikit-learn does it


import numpy as np
from dataclasses import dataclass


@dataclass
class Metric:
    gt: np.ndarray
    est: np.ndarray
    axis: tuple
    # fwd: Any = None
    metric_fn: Any = None
    label: str = ""

    def __call__(
        self,
    ) -> Any:
        return self.metric()

    def metric(self) -> Any:
        return self.metric_fn(gt=self.gt, est=self.est, axis=self.axis)


from . import loss_functions

import functools

@functools.lru_cache()
def cached_all(all,key):
    return all[key]()

class ReconstructionMetrics2D(Metric):
    axis = (-2, -1)

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        # self.loglikelihood = Loglikelihood(axis=self.axis, **kwargs)
        # self.poisson = Poisson(axis=self.axis, **kwargs)
        # self.crossentropy = CrossEntropy(axis=self.axis, **kwargs)
        # self.ncc = NCC(axis=self.axis, **kwargs)
        # self.kl = KLNoiselessSignal(axis=self.axis, **kwargs)
        # self.loglikelihood = Loglikelihood(axis=self.axis, **self.kwargs)
        # self.poisson = Poisson(axis=self.axis, **self.kwargs)
        # self.crossentropy = CrossEntropy(axis=self.axis, **self.kwargs)
        # self.ncc = NCC(axis=self.axis, **self.kwargs)
        # self.kl = KLNoiselessSignal(axis=self.axis, **self.kwargs)

        # self.all = {
        #     "loglikelihood": self.loglikelihood,
        #     "poisson": self.poisson,
        #     "ncc": self.ncc,
        #     "crossentropy": self.crossentropy,
        #     "kl": self.kl,
        #     }
        
    @functools.cached_property
    def loglikelihood(self):
        return Loglikelihood(axis=self.axis, **self.kwargs)()
    
    @functools.cached_property
    def poisson(self):
        return Poisson(axis=self.axis, **self.kwargs)()
    
    @functools.cached_property
    def crossentropy(self):
        return CrossEntropy(axis=self.axis, **self.kwargs)()
    
    @functools.cached_property
    def ncc(self):
        return NCC(axis=self.axis, **self.kwargs)()
    
    @functools.cached_property
    def kl(self):
        return KLNoiselessSignal(axis=self.axis, **self.kwargs)()
    
    
    def __call__(self,key):
        return self[key]
    
    def __getitem__(self, key):
        return getattr(self,key)





class Metric2D(Metric):
    axis = (-2, -1)

    def __init__(
        self,
        est,
        gt,
    ):
        super().__init__(est=est, gt=gt, axis=self.axis)
        self.est = est
        self.gt = gt

    def to_df(self):
        return pd.DataFrame(self.metric(), index=["obj", "V", "T"]).rename_axis(
            "splitting"
        )


class Loglikelihood(Metric):
    def metric(self):
        return loss_functions.loglikelihood(
            est=self.est,
            gt=self.gt,
            axis=self.axis,
        )


class Poisson(Metric):
    def metric(self):
        return loss_functions.poisson(
            est=self.est,
            gt=self.gt,
            axis=self.axis,
        )


class NCC(Metric):
    def metric(self):
        return loss_functions.ncc(
            est=self.est,
            gt=self.gt,
            axis=self.axis,
        )


class CrossEntropy(Metric):
    def metric(self):
        return loss_functions.crossentropy(
            est=self.est,
            gt=self.gt,
            axis=self.axis,
        )


class KLNoiselessSignal(Metric):
    def metric(self):
        return loss_functions.kl(
            est=self.est,
            gt=self.gt,
            axis=self.axis,
        )
