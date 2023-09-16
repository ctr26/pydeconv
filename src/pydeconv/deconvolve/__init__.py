# Expanded the modified richardson lucy equation to the first two components.
from .richardson_lucy import RichardsonLucy


def deconvolve(method="rl", psf=None, max_iterations=25, *args, **kwargs):
    return DeconvolverFactory(method=method)(
        psf, max_iterations, *args, **kwargs
    )


class DeconvolverFactory:
    def __init__(
        self,
        method="rl",
        *args,
        **kwargs,
    ):
        self.method = method

    def __call__(self, *args, **kwargs):
        return getattr("self", self.method)(*args, **kwargs)

    def rl(self, psf, max_iterations, *args, **kwargs):
        return RichardsonLucy(
            psf=psf, max_iterations=max_iterations, *args, **kwargs
        )

    def richardson_lucy(self, *args, **kwargs):
        return self.rl(*args, **kwargs)

    def weiner(self, psf, max_iterations, *args, **kwargs):
        return NotImplementedError("Weiner not implemented yet")

    def total_variation(self, psf, max_iterations, *args, **kwargs):
        return NotImplementedError("Total variation not implemented yet")

    def tv(self, *args, **kwargs):
        return self.total_variation(*args, **kwargs)
