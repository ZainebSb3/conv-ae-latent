#!/usr/bin/env bash
# Copies the figures used in the README from runs/ (not versioned) to assets/ (versioned).
set -euo pipefail
mkdir -p assets
cp runs/cifar10_d64/reconstructions.png assets/
cp runs/cifar10_d64/tsne.png assets/
cp runs/cifar10_d64/anomaly_svhn.png assets/
cp runs/compression/rate_distortion.png assets/
ls -lh assets/
