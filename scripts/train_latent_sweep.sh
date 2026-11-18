#!/usr/bin/env bash
# Trains the AE for several latent sizes (same config, same seed).
set -euo pipefail
for d in "$@"; do
  python -m convae.train --config configs/cifar10_d64.yaml --set model.latent_dim="$d" run_name="cifar10_d$d"
  python -m convae.evaluate --run "runs/cifar10_d$d"
done
