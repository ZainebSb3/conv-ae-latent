import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms


def _tf(img_size: int):
    # Uniquement ToTensor -> pixels dans [0, 1]. Pas de normalisation moyenne/écart-type :
    # la sortie sigmoïde et le PSNR/SSIM supposent [0, 1].
    t = [transforms.ToTensor()]
    if img_size is not None:
        t.insert(0, transforms.Resize(img_size))
    return transforms.Compose(t)


def get_datasets(name: str, root: str = "data", img_size: int = 32, val_fraction: float = 0.1, seed: int = 0):
    """Renvoie (train, val, test). val est extrait du train avec une permutation fixée par la graine."""
    tf = _tf(img_size)
    if name == "cifar10":
        full = datasets.CIFAR10(root, train=True, download=True, transform=tf)
        test = datasets.CIFAR10(root, train=False, download=True, transform=tf)
    elif name == "stl10":
        full = datasets.STL10(root, split="train", download=True, transform=tf)
        test = datasets.STL10(root, split="test", download=True, transform=tf)
    elif name == "fake":  # pour vérifier le pipeline sans téléchargement
        full = datasets.FakeData(512, (3, img_size, img_size), 10, transform=transforms.ToTensor())
        test = datasets.FakeData(128, (3, img_size, img_size), 10, transform=transforms.ToTensor(), random_offset=1)
    else:
        raise ValueError(f"dataset inconnu : {name}")

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(full), generator=g).tolist()
    n_val = int(len(full) * val_fraction)
    return Subset(full, perm[n_val:]), Subset(full, perm[:n_val]), test
