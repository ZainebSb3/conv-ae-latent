# conv-ae-latent

Autoencodeur convolutionnel en PyTorch et réutilisation de son espace latent (encodeur gelé) pour la classification par sonde linéaire, la détection d'anomalies et la compression d'images.

Ce projet personnel prolonge mon travail sur les autoencodeurs pour la détection d'anomalies sur des données réseau (HAIDS-PoA). Ici, je passe aux images.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Entraînement

```bash
python -m convae.train --config configs/cifar10_d64.yaml
# autre taille de latent
python -m convae.train --config configs/cifar10_d64.yaml --set model.latent_dim=128 run_name=cifar10_d128
```

Sorties dans `runs/<run_name>/` : `config.json`, `metrics.csv` (MSE et PSNR par époque), `best.pt` (meilleur modèle en validation).

## Tests

```bash
pytest
```

## Résultats

À venir. Seules des valeurs mesurées par les scripts du dépôt figureront ici.
