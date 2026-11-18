import pytest
import torch
import torch.nn.functional as F

from convae.model import ConvAutoencoder
from convae.utils import load_config, set_seed


@pytest.mark.parametrize("d", [16, 64, 256])
def test_shapes_and_range(d):
    model = ConvAutoencoder(latent_dim=d).eval()
    x = torch.rand(4, 3, 32, 32)
    x_hat, z = model(x)
    assert z.shape == (4, d)
    assert x_hat.shape == x.shape
    assert x_hat.min() >= 0 and x_hat.max() <= 1  # sigmoid output


def test_seed_reproducible():
    set_seed(123)
    a = ConvAutoencoder(32)
    set_seed(123)
    b = ConvAutoencoder(32)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)


def test_can_overfit_tiny_batch():
    # If the model cannot memorize 4 simple images, something is broken (gradients, shapes, ...).
    set_seed(0)
    colors = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=torch.float32)
    x = colors[:, :, None, None].expand(4, 3, 32, 32).contiguous()
    model = ConvAutoencoder(16)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    first = None
    for _ in range(150):
        loss = F.mse_loss(model(x)[0], x)
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1 * first


def test_config_override(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("model:\n  latent_dim: 64\nrun_name: a\n")
    cfg = load_config(str(p), ["model.latent_dim=128", "run_name=b"])
    assert cfg["model"]["latent_dim"] == 128 and cfg["run_name"] == "b"
