"""Évaluation finale de la reconstruction sur le jeu de TEST (une seule fois, avec best.pt).

python -m convae.evaluate --run runs/cifar10_d64
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from convae.data import get_datasets
from convae.features import load_model
from convae.train import evaluate
from convae.utils import get_device, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n_show", type=int, default=8)
    args = ap.parse_args()

    device = get_device()
    model, cfg = load_model(args.run, device)
    set_seed(cfg["seed"])
    d = cfg["data"]
    _, _, test = get_datasets(d["name"], d["root"], d["img_size"], d["val_fraction"], cfg["seed"])

    mse, psnr_img = evaluate(model, DataLoader(test, 512), device)
    res = {"test_mse": mse, "test_psnr_per_image_mean": psnr_img,
           "test_psnr_of_mean_mse": float(10 * torch.log10(torch.tensor(1.0 / mse)))}
    Path(args.run, "eval_test.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))

    # Grille : ligne du haut = originaux, ligne du bas = reconstructions (premières images du test)
    x = torch.stack([test[i][0] for i in range(args.n_show)]).to(device)
    with torch.no_grad():
        x_hat, _ = model(x)
    save_image(torch.cat([x, x_hat]), Path(args.run, "reconstructions.png"), nrow=args.n_show, padding=2)
    print(f"-> {args.run}/eval_test.json, {args.run}/reconstructions.png")


if __name__ == "__main__":
    main()
