"""Entraînement de l'autoencodeur.

python -m convae.train --config configs/cifar10_d64.yaml
python -m convae.train --config configs/cifar10_d64.yaml --set model.latent_dim=128 run_name=cifar10_d128
"""
import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from convae.data import get_datasets
from convae.model import ConvAutoencoder
from convae.utils import get_device, load_config, psnr, seed_worker, set_seed


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    se, n_pix, psnr_sum, n_img = 0.0, 0, 0.0, 0
    for x, _ in loader:
        x = x.to(device)
        x_hat, _ = model(x)
        err = (x_hat - x) ** 2
        se += err.sum().item()
        n_pix += err.numel()
        psnr_sum += psnr(err.flatten(1).mean(1)).sum().item()  # PSNR par image puis moyenne
        n_img += x.size(0)
    return se / n_pix, psnr_sum / n_img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="surcharges cle.sous_cle=valeur")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    set_seed(cfg["seed"])
    device = get_device()
    out = Path("runs") / cfg["run_name"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    d, t = cfg["data"], cfg["train"]
    train_ds, val_ds, _ = get_datasets(d["name"], d["root"], d["img_size"], d["val_fraction"], cfg["seed"])
    g = torch.Generator().manual_seed(cfg["seed"])
    train_dl = DataLoader(train_ds, t["batch_size"], shuffle=True, num_workers=t["num_workers"],
                          worker_init_fn=seed_worker, generator=g, pin_memory=device.type == "cuda")
    val_dl = DataLoader(val_ds, 512, shuffle=False, num_workers=t["num_workers"])

    m = cfg["model"]
    model = ConvAutoencoder(m["latent_dim"], tuple(m["channels"]), d["img_size"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device} params={n_params:,} train={len(train_ds)} val={len(val_ds)}")

    best = float("inf")
    with open(out / "metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_mse", "val_mse", "val_psnr", "seconds"])
        for ep in range(1, t["epochs"] + 1):
            model.train()
            t0, tot, n = time.time(), 0.0, 0
            for x, _ in train_dl:  # les labels sont ignorés : apprentissage non supervisé
                x = x.to(device, non_blocking=True)
                x_hat, _ = model(x)
                loss = F.mse_loss(x_hat, x)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                tot += loss.item() * x.size(0)
                n += x.size(0)
            val_mse, val_psnr = evaluate(model, val_dl, device)
            dt = time.time() - t0
            w.writerow([ep, f"{tot / n:.6f}", f"{val_mse:.6f}", f"{val_psnr:.3f}", f"{dt:.1f}"])
            f.flush()
            print(f"ep {ep:3d}  train_mse {tot / n:.5f}  val_mse {val_mse:.5f}  val_psnr {val_psnr:.2f} dB  ({dt:.0f}s)")
            if val_mse < best:  # on garde le meilleur modèle sur la validation, pas le dernier
                best = val_mse
                torch.save({"model": model.state_dict(), "config": cfg, "epoch": ep}, out / "best.pt")
    print(f"meilleur val_mse={best:.5f} -> {out / 'best.pt'}")


if __name__ == "__main__":
    main()
