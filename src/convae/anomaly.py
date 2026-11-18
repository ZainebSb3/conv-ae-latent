"""Anomaly / out-of-distribution detection with the frozen AE.

Normal = CIFAR-10 test (same distribution as training), anomaly = SVHN test.
Three scores (higher = more anomalous):
  - recon: per-image reconstruction MSE
  - knn  : mean distance to the k nearest train latents (standardized latents)
  - maha : Mahalanobis distance to the train latent cloud (Ledoit-Wolf covariance)
Latent scores are also computed with the random encoder, as a baseline.

python -m convae.anomaly --run runs/cifar10_d64 --ood svhn
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from convae.data import get_datasets, get_ood_dataset
from convae.features import extract_latents, load_model
from convae.utils import get_device, set_seed


@torch.no_grad()
def recon_errors(model, dataset, device, batch_size=512):
    errs = []
    for x, _ in DataLoader(dataset, batch_size, shuffle=False):
        x = x.to(device)
        errs.append(((model(x)[0] - x) ** 2).flatten(1).mean(1).cpu())
    return torch.cat(errs).numpy()


def fpr_at_tpr(y, s, tpr_target=0.95):
    """False positive rate (normal images flagged) when 95% of anomalies are detected."""
    fpr, tpr, _ = roc_curve(y, s)
    return float(fpr[np.searchsorted(tpr, tpr_target, side="left")])


def score_metrics(s_normal, s_anom):
    y = np.r_[np.zeros(len(s_normal)), np.ones(len(s_anom))]  # 1 = anomaly
    s = np.r_[s_normal, s_anom]
    return {"auroc": float(roc_auc_score(y, s)), "fpr_at_95tpr": fpr_at_tpr(y, s)}


def latent_scores(Ztr, Zn, Za, k):
    sc = StandardScaler().fit(Ztr)  # train statistics only
    Ztr, Zn, Za = sc.transform(Ztr), sc.transform(Zn), sc.transform(Za)
    nn = NearestNeighbors(n_neighbors=k).fit(Ztr)
    knn = lambda Z: nn.kneighbors(Z)[0].mean(1)
    lw = LedoitWolf().fit(Ztr)  # shrunk covariance: more stable than the empirical one
    return {"knn": (knn(Zn), knn(Za)), "maha": (lw.mahalanobis(Zn), lw.mahalanobis(Za))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ood", default="svhn")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    device = get_device()
    model, cfg = load_model(args.run, device)
    set_seed(cfg["seed"])
    d = cfg["data"]
    train, _, normal = get_datasets(d["name"], d["root"], d["img_size"], d["val_fraction"], cfg["seed"])
    anom = get_ood_dataset(args.ood if d["name"] != "fake" else "fake", d["root"], d["img_size"])
    print(f"normal={len(normal)} ({d['name']} test)  anomalies={len(anom)} ({args.ood} test)")

    results, scores = {}, {}
    scores["recon"] = (recon_errors(model, normal, device), recon_errors(model, anom, device))

    for tag, m in [("", model), ("random_", load_model(args.run, device, pretrained=False)[0])]:
        Ztr, _ = extract_latents(m, train, device)
        Zn, _ = extract_latents(m, normal, device)
        Za, _ = extract_latents(m, anom, device)
        for name, pair in latent_scores(Ztr, Zn, Za, args.k).items():
            scores[tag + name] = pair

    for name, (sn, sa) in scores.items():
        results[name] = score_metrics(sn, sa)
        results[name]["mean_normal"], results[name]["mean_anomaly"] = float(sn.mean()), float(sa.mean())
        print(f"{name:12s} AUROC={results[name]['auroc']:.4f}  FPR@95TPR={results[name]['fpr_at_95tpr']:.4f}"
              f"  mean normal={sn.mean():.4g}  anomaly={sa.mean():.4g}")

    out = Path(args.run)
    (out / f"anomaly_{args.ood}.json").write_text(json.dumps(results, indent=2))

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    for ax, name in zip(axes, ["recon", "knn", "maha"]):
        sn, sa = scores[name]
        bins = np.histogram_bin_edges(np.r_[sn, sa], bins=60)
        ax.hist(sn, bins, alpha=0.6, density=True, label=f"{d['name']} (normal)")
        ax.hist(sa, bins, alpha=0.6, density=True, label=f"{args.ood} (anomaly)")
        ax.set_title(f"{name}  AUROC={results[name]['auroc']:.3f}")
        ax.set_xlabel("score")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out / f"anomaly_{args.ood}.png", dpi=120)
    print(f"-> {out}/anomaly_{args.ood}.json, {out}/anomaly_{args.ood}.png")


if __name__ == "__main__":
    main()
