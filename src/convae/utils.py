import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    """Fixe toutes les sources d'aléatoire. Le déterminisme GPU complet n'est pas garanti
    (certaines opérations cuDNN), d'où benchmark=False / deterministic=True."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    # Chaque worker du DataLoader reçoit une graine dérivée de la graine torch.
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def _cast(v: str):
    for f in (int, float):
        try:
            return f(v)
        except ValueError:
            pass
    return {"true": True, "false": False}.get(v.lower(), v)


def load_config(path: str, overrides: list[str] | None = None) -> dict:
    """Charge un YAML puis applique des surcharges 'model.latent_dim=128'."""
    cfg = yaml.safe_load(Path(path).read_text())
    for ov in overrides or []:
        key, val = ov.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node[p]
        node[leaf] = _cast(val)
    return cfg


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def psnr(mse: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """PSNR en dB, pour des images dans [0, 1]."""
    return 10 * torch.log10(max_val**2 / mse.clamp_min(1e-12))
