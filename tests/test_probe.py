import numpy as np
import torch
from torch.utils.data import TensorDataset

from convae.features import extract_latents
from convae.model import ConvAutoencoder
from convae.probe import linear_probe


def test_extract_latents_shapes():
    ds = TensorDataset(torch.rand(10, 3, 32, 32), torch.arange(10))
    Z, y = extract_latents(ConvAutoencoder(24).eval(), ds, torch.device("cpu"), batch_size=4)
    assert Z.shape == (10, 24) and y.shape == (10,)


def test_linear_probe_separable():
    # Two well-separated clouds: the probe should reach ~100%.
    rng = np.random.default_rng(0)
    def make(n):
        y = rng.integers(0, 2, n)
        return rng.normal(0, 1, (n, 5)) + 6 * y[:, None], y
    r = linear_probe(*make(200), *make(100), *make(100))
    assert r["test_acc"] > 0.98
