# %% Import required modules
import numpy as np
from scipy import special, fft, interpolate
import logging

logger = logging.getLogger("dev")
logger.setLevel(logging.INFO)
# %% ---------------------------------------------------------------------------
# %% Some functions that are required for the code.

from pydeconv import utils


def create_spokes(numPixel, midPos):
    r = utils.rr(numPixel, numPixel)
    phi = utils.phi(numPixel)
    obj = np.zeros(numPixel)
    obj[r < (0.5 * np.max(r))] = 1.0
    obj[np.mod(phi + np.pi, 2 * np.pi / 18) < 2 * np.pi / 18 / 2] = 0.0
    return obj


def create_test_target(numPixel, midPos):
    N = numPixel[0]
    x = np.arange(N) / N - 0.5
    y = np.arange(N) / N - 0.5
    aa = np.zeros((N, N))
    aa[:: int(N / 24), :] = 1

    X, Y = np.meshgrid(x, y)
    aa *= Y + 0.5
    R = np.sqrt(X**2 + Y**2)
    f0 = 4
    k = 50
    a = 0.5 * (1 + np.cos(np.pi * 2 * (f0 * R + k * R**2 / 2)))
    a[: int(N / 2), :][R[: int(N / 2), :] < 0.4] = 0
    a[: int(N / 2), :][R[: int(N / 2), :] < 0.3] = 1
    a[: int(N / 2), :][R[: int(N / 2), :] < 0.15] = 0

    a[int(N * 3 / 4) :, int(N * 3 / 4) :] = 0
    for l in np.arange(0, 2 * k):
        ind = np.random.randint(int(N * 1 / 4) - 1, size=(2))
        d = np.random.randint(2)
        a[
            ind[0] + int(N * 3 / 4) - d + 1 : ind[0] + int(N * 3 / 4) + d + 1,
            ind[1] + int(N * 3 / 4) - d + 1 : ind[1] + int(N * 3 / 4) + d + 1,
        ] = 1
    aa[:, int(N / 4) :] = a[:, int(N / 4) :]
    aa[: int(N / 32), :] = 0
    aa[N - int(N / 48) :, :] = 0
    aa[:, : int(N / 48)] = 0
    aa[:, N - int(N / 32) :] = 0
    obj = aa
    return obj


def create_points_random(numPixel, midPos, seed=42):
    obj = np.zeros(numPixel)
    pos = np.random.randint(
        midPos - np.floor(np.min(numPixel[0]) / 3).astype(int),
        midPos + np.floor(np.min(numPixel[1]) / 3).astype(int),
        size=(100, 2),
    )
    obj[pos[:, 0], pos[:, 1]] = np.random.rand(100) + 1.0
    return obj


# object_mapping = {
#     "spokes":create_spokes,
#     "test_target":create_test_target,
#     "points_random":create_points_random
# }


class CreateObject:
    def __init__(self, numPixel, midPos):
        self.numPixel = numPixel
        self.midPos = midPos

    def spokes(self):
        return create_spokes(self.numPixel, self.midPos)

    def test_target(self):
        return create_test_target(self.numPixel, self.midPos)

    def points_random(self):
        return create_points_random(self.numPixel, self.midPos)


def create_object(obj_name, numPixel, midPos, seed=42):
    obj = CreateObject(numPixel, midPos)
    return getattr(obj, obj_name)()


# def CreateObject(obj_name, numPixel, midPos):
#     x = xx(numPixel, numPixel)
#     x = yy(numPixel, numPixel)
#     r = rr(numPixel, numPixel)
#     phi = phiphi(numPixel)

#     obj = np.zeros(numPixel)
#     if obj_name == "spokes":
#         obj[r < (0.5 * np.max(r))] = 1.0
#         obj[np.mod(phi + np.pi, 2 * np.pi / 18) < 2 * np.pi / 18 / 2] = 0.0
#     elif obj_name == "test_target":
#         N = numPixel[0]
#         x = np.arange(N) / N - 0.5
#         y = np.arange(N) / N - 0.5
#         aa = np.zeros((N, N))
#         aa[:: int(N / 24), :] = 1

#         X, Y = np.meshgrid(x, y)
#         aa *= Y + 0.5
#         R = np.sqrt(X**2 + Y**2)
#         f0 = 4
#         k = 50
#         a = 0.5 * (1 + np.cos(np.pi * 2 * (f0 * R + k * R**2 / 2)))
#         a[: int(N / 2), :][R[: int(N / 2), :] < 0.4] = 0
#         a[: int(N / 2), :][R[: int(N / 2), :] < 0.3] = 1
#         a[: int(N / 2), :][R[: int(N / 2), :] < 0.15] = 0

#         a[int(N * 3 / 4) :, int(N * 3 / 4) :] = 0
#         for l in np.arange(0, 2 * k):
#             ind = np.random.randint(int(N * 1 / 4) - 1, size=(2))
#             d = np.random.randint(2)
#             a[
#                 ind[0] + int(N * 3 / 4) - d + 1 : ind[0] + int(N * 3 / 4) + d + 1,
#                 ind[1] + int(N * 3 / 4) - d + 1 : ind[1] + int(N * 3 / 4) + d + 1,
#             ] = 1
#         aa[:, int(N / 4) :] = a[:, int(N / 4) :]
#         aa[: int(N / 32), :] = 0
#         aa[N - int(N / 48) :, :] = 0
#         aa[:, : int(N / 48)] = 0
#         aa[:, N - int(N / 32) :] = 0
#         obj = aa
#     elif obj_name == "points_random":
#         pos = np.random.randint(
#             midPos - np.floor(np.min(numPixel[0]) / 3).astype(int),
#             midPos + np.floor(np.min(numPixel[1]) / 3).astype(int),
#             size=(100, 2),
#         )
#         obj[pos[:, 0], pos[:, 1]] = np.random.rand(100) + 1.0

#     return obj
