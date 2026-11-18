"""Compression : AE (latent quantifié sur 8 bits) contre JPEG, sur le test CIFAR-10.

Débit en bits par pixel (bpp) = bits stockés / (H*W). Qualité : PSNR et SSIM par image, moyennés.
JPEG est mesuré de deux façons :
  - jpeg_single : chaque image 32x32 encodée seule (en-tête compris, comme un vrai fichier)
  - jpeg_mosaic : les 10 000 images assemblées en une mosaïque encodée une fois ; l'en-tête est
    amorti. 32 est multiple des blocs JPEG (8x8, MCU 16x16 en 4:2:0) : aucun bloc ne mélange deux images.
Non compté pour l'AE : la taille du décodeur (poids), et le fait qu'il est spécialisé sur CIFAR-10.

python -m convae.compression --runs runs/cifar10_d16 runs/cifar10_d32 runs/cifar10_d64 runs/cifar10_d128 runs/cifar10_d256
"""
import argparse
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from convae.data import get_datasets
from convae.features import extract_latents, load_model
from convae.utils import get_device, set_seed

JPEG_QUALITIES = [5, 10, 20, 30, 50, 70, 90]


def quality(ref: np.ndarray, rec: np.ndarray):
    """ref, rec : (N, H, W, 3) float dans [0, 1]. Renvoie PSNR et SSIM moyens par image."""
    p = [peak_signal_noise_ratio(a, b, data_range=1.0) for a, b in zip(ref, rec)]
    s = [structural_similarity(a, b, data_range=1.0, channel_axis=-1) for a, b in zip(ref, rec)]
    return float(np.mean(p)), float(np.mean(s))


def quantize(z: np.ndarray, lo: np.ndarray, hi: np.ndarray, bits: int):
    """Quantification uniforme par dimension, bornes apprises sur le train. Renvoie les valeurs déquantifiées."""
    levels = 2**bits - 1
    q = np.round((np.clip(z, lo, hi) - lo) / (hi - lo) * levels)
    return (q / levels * (hi - lo) + lo).astype(np.float32)


def jpeg_roundtrip(img_u8: np.ndarray, q: int):
    buf = io.BytesIO()
    Image.fromarray(img_u8).save(buf, format="JPEG", quality=q)
    n = buf.tell()
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB")), n


def to_u8(x: np.ndarray):
    return np.round(x * 255).clip(0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--out", default="runs/compression")
    args = ap.parse_args()
    device = get_device()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    # --- AE ---
    for run in args.runs:
        model, cfg = load_model(run, device)
        set_seed(cfg["seed"])
        d = cfg["data"]
        train, _, test = get_datasets(d["name"], d["root"], d["img_size"], d["val_fraction"], cfg["seed"])
        Ztr, _ = extract_latents(model, train, device)
        Zte, _ = extract_latents(model, test, device)
        lo, hi = Ztr.min(0), Ztr.max(0)
        X = torch.stack([test[i][0] for i in range(len(test))])  # (N, 3, H, W)
        ref = X.permute(0, 2, 3, 1).numpy()
        n_pix = ref.shape[1] * ref.shape[2]
        dim = cfg["model"]["latent_dim"]
        for codec, Z, bits in [("ae_float32", Zte, 32), (f"ae_int{args.bits}", quantize(Zte, lo, hi, args.bits), args.bits)]:
            with torch.no_grad():
                rec = torch.cat([model.decode(torch.from_numpy(Z[i:i + 1000]).to(device)).cpu()
                                 for i in range(0, len(Z), 1000)])
            # Arrondi à 8 bits/canal comme un vrai fichier image, pour comparer à JPEG à armes égales
            rec = to_u8(rec.permute(0, 2, 3, 1).numpy()) / 255.0
            psnr, ssim = quality(ref, rec)
            rows.append({"codec": codec, "param": dim, "bpp": dim * bits / n_pix, "psnr": psnr, "ssim": ssim})
            print(f"{codec:11s} d={dim:4d}  bpp={rows[-1]['bpp']:.3f}  PSNR={psnr:.2f}  SSIM={ssim:.4f}")

    # --- JPEG ---
    ref_u8 = to_u8(ref)
    ref = ref_u8 / 255.0  # référence identique pour tous les codecs
    N, H, W, _ = ref_u8.shape
    side = int(np.ceil(np.sqrt(N)))
    mosaic = np.zeros((side * H, side * W, 3), np.uint8)
    for i in range(N):
        r, c = divmod(i, side)
        mosaic[r * H:(r + 1) * H, c * W:(c + 1) * W] = ref_u8[i]
    for q in JPEG_QUALITIES:
        dec, sizes = zip(*(jpeg_roundtrip(im, q) for im in ref_u8))
        psnr, ssim = quality(ref, np.stack(dec) / 255.0)
        rows.append({"codec": "jpeg_single", "param": q, "bpp": 8 * float(np.mean(sizes)) / (H * W), "psnr": psnr, "ssim": ssim})
        print(f"jpeg_single q={q:3d}  bpp={rows[-1]['bpp']:.3f}  PSNR={psnr:.2f}  SSIM={ssim:.4f}")

        mdec, msize = jpeg_roundtrip(mosaic, q)
        tiles = np.stack([mdec[(i // side) * H:(i // side + 1) * H, (i % side) * W:(i % side + 1) * W] for i in range(N)])
        psnr, ssim = quality(ref, tiles / 255.0)
        rows.append({"codec": "jpeg_mosaic", "param": q, "bpp": 8 * msize / (N * H * W), "psnr": psnr, "ssim": ssim})
        print(f"jpeg_mosaic q={q:3d}  bpp={rows[-1]['bpp']:.3f}  PSNR={psnr:.2f}  SSIM={ssim:.4f}")

    (out / "compression.json").write_text(json.dumps(rows, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ["psnr", "ssim"]):
        for codec, style in [(f"ae_int{args.bits}", "o-"), ("jpeg_single", "s--"), ("jpeg_mosaic", "^--")]:
            pts = sorted((r["bpp"], r[metric], r["param"]) for r in rows if r["codec"] == codec)
            if not pts:
                continue
            bx, by, labels = zip(*pts)
            ax.plot(bx, by, style, label=codec)
            if codec.startswith("ae"):
                for x_, y_, l_ in pts:
                    ax.annotate(f"d={l_}", (x_, y_), textcoords="offset points", xytext=(4, -10), fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("bits per pixel (log)")
        ax.set_ylabel(metric.upper())
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out / "rate_distortion.png", dpi=120)
    print(f"-> {out}/compression.json, {out}/rate_distortion.png")


if __name__ == "__main__":
    main()
