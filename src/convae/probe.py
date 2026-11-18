"""Sonde linéaire : régression logistique sur des représentations figées.

Compare, à protocole identique :
  - ae_pretrained : latents de l'encodeur entraîné (dim d)
  - ae_random     : même architecture, poids aléatoires (l'apprentissage apporte-t-il quelque chose ?)
  - pca_d         : ACP des pixels à la même dimension d (le non-linéaire bat-il le linéaire ?)
  - pixels        : pixels bruts, 3072 dims (optionnel, plus lent)

python -m convae.probe --run runs/cifar10_d64
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from convae.data import get_datasets
from convae.features import extract_latents, extract_pixels, load_model
from convae.utils import get_device, set_seed

C_GRID = [0.01, 0.1, 1.0, 10.0]


def linear_probe(Xtr, ytr, Xval, yval, Xte, yte, seed=0):
    """Standardisation apprise sur le train uniquement ; C choisi sur la validation ; score final sur le test."""
    sc = StandardScaler().fit(Xtr)
    Xtr, Xval, Xte = sc.transform(Xtr), sc.transform(Xval), sc.transform(Xte)
    best = None
    for C in C_GRID:
        clf = LogisticRegression(C=C, max_iter=2000, random_state=seed).fit(Xtr, ytr)
        acc = clf.score(Xval, yval)
        if best is None or acc > best[1]:
            best = (C, acc, clf)
    C, val_acc, clf = best
    return {"C": C, "val_acc": val_acc, "test_acc": clf.score(Xte, yte), "dim": Xtr.shape[1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--skip_pixels", action="store_true", help="saute la référence pixels bruts (lente)")
    args = ap.parse_args()

    device = get_device()
    model, cfg = load_model(args.run, device, pretrained=True)
    set_seed(cfg["seed"])
    d = cfg["data"]
    tr, va, te = get_datasets(d["name"], d["root"], d["img_size"], d["val_fraction"], cfg["seed"])
    results = {}

    def run(name, Xtr, ytr, Xva, yva, Xte, yte):
        t0 = time.time()
        r = linear_probe(Xtr, ytr, Xva, yva, Xte, yte, cfg["seed"])
        r["seconds"] = round(time.time() - t0, 1)
        results[name] = r
        print(f"{name:14s} dim={r['dim']:5d}  C={r['C']:<5}  val_acc={r['val_acc']:.4f}  test_acc={r['test_acc']:.4f}")

    run("ae_pretrained", *extract_latents(model, tr, device), *extract_latents(model, va, device),
        *extract_latents(model, te, device))

    rand, _ = load_model(args.run, device, pretrained=False)
    run("ae_random", *extract_latents(rand, tr, device), *extract_latents(rand, va, device),
        *extract_latents(rand, te, device))

    Ptr, ytr = extract_pixels(tr)
    Pva, yva = extract_pixels(va)
    Pte, yte = extract_pixels(te)
    k = cfg["model"]["latent_dim"]
    pca = PCA(n_components=k, random_state=cfg["seed"]).fit(Ptr)  # ACP ajustée sur le train seulement
    run(f"pca_{k}", pca.transform(Ptr), ytr, pca.transform(Pva), yva, pca.transform(Pte), yte)

    if not args.skip_pixels:
        run("pixels", Ptr, ytr, Pva, yva, Pte, yte)

    out = Path(args.run) / "probe.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
