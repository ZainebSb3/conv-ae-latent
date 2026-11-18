"""Chargement d'un modèle entraîné et extraction des latents avec l'encodeur gelé."""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from convae.model import ConvAutoencoder


def load_model(run_dir: str, device, pretrained: bool = True):
    """Recharge l'AE d'un run. pretrained=False -> même architecture, poids aléatoires (référence)."""
    ckpt = torch.load(Path(run_dir) / "best.pt", map_location=device, weights_only=False)
    cfg = ckpt["config"]
    m = cfg["model"]
    if not pretrained:
        torch.manual_seed(cfg["seed"])
    model = ConvAutoencoder(m["latent_dim"], tuple(m["channels"]), cfg["data"]["img_size"])
    if pretrained:
        model.load_state_dict(ckpt["model"])
    model.to(device).eval()  # eval() : BatchNorm utilise ses statistiques apprises, pas celles du batch
    for p in model.parameters():
        p.requires_grad_(False)  # encodeur gelé
    return model, cfg


@torch.no_grad()
def extract_latents(model, dataset, device, batch_size: int = 512):
    """Renvoie Z (N, d) et y (N,) en numpy."""
    zs, ys = [], []
    for x, y in DataLoader(dataset, batch_size, shuffle=False):
        zs.append(model.encode(x.to(device)).cpu())
        ys.append(torch.as_tensor(y))
    return torch.cat(zs).numpy(), torch.cat(ys).numpy()


def extract_pixels(dataset, batch_size: int = 1024):
    """Pixels aplatis (N, 3*H*W), pour les références."""
    xs, ys = [], []
    for x, y in DataLoader(dataset, batch_size, shuffle=False):
        xs.append(x.flatten(1))
        ys.append(torch.as_tensor(y))
    return torch.cat(xs).numpy().astype(np.float32), torch.cat(ys).numpy()
