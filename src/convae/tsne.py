"""2-D t-SNE of test-set latents, colored by class: trained encoder vs random encoder.

t-SNE is only a visualization: it preserves local neighborhoods, not global distances,
so cluster sizes and inter-cluster gaps are not interpretable. The quantitative
evidence is the linear probe; this figure is a qualitative complement.

python -m convae.tsne --run runs/cifar10_d64
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Subset

from convae.data import get_datasets
from convae.features import extract_latents, load_model
from convae.utils import get_device, set_seed

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def embed_2d(Z: np.ndarray, seed: int, perplexity: float = 30.0) -> np.ndarray:
    Z = StandardScaler().fit_transform(Z)
    return TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=seed).fit_transform(Z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=5000, help="number of test images (t-SNE is O(n log n) but slow)")
    ap.add_argument("--perplexity", type=float, default=30.0)
    args = ap.parse_args()

    device = get_device()
    model, cfg = load_model(args.run, device)
    rand, _ = load_model(args.run, device, pretrained=False)
    set_seed(cfg["seed"])
    d = cfg["data"]
    _, _, test = get_datasets(d["name"], d["root"], d["img_size"], d["val_fraction"], cfg["seed"])
    idx = np.random.default_rng(cfg["seed"]).choice(len(test), size=min(args.n, len(test)), replace=False)
    sub = Subset(test, idx.tolist())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, (title, m) in zip(axes, [("trained encoder", model), ("random encoder", rand)]):
        Z, y = extract_latents(m, sub, device)
        E = embed_2d(Z, cfg["seed"], args.perplexity)
        for c in range(10):
            pts = E[y == c]
            ax.scatter(pts[:, 0], pts[:, 1], s=3, alpha=0.6, label=CIFAR10_CLASSES[c])
        ax.set_title(f"t-SNE of test latents, d={cfg['model']['latent_dim']}, {title}")
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].legend(markerscale=4, fontsize=8, loc="best")
    fig.tight_layout()
    out = Path(args.run) / "tsne.png"
    fig.savefig(out, dpi=120)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
