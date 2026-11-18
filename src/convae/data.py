import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms


def _tf(img_size: int):
    # ToTensor only -> pixels in [0, 1]. No mean/std normalization:
    # the sigmoid output and PSNR/SSIM assume [0, 1].
    t = [transforms.ToTensor()]
    if img_size is not None:
        t.insert(0, transforms.Resize(img_size))
    return transforms.Compose(t)


def get_datasets(name: str, root: str = "data", img_size: int = 32, val_fraction: float = 0.1, seed: int = 0):
    """Returns (train, val, test). val is split off train with a seeded permutation."""
    tf = _tf(img_size)
    if name == "cifar10":
        full = datasets.CIFAR10(root, train=True, download=True, transform=tf)
        test = datasets.CIFAR10(root, train=False, download=True, transform=tf)
    elif name == "stl10":
        full = datasets.STL10(root, split="train", download=True, transform=tf)
        test = datasets.STL10(root, split="test", download=True, transform=tf)
    elif name == "fake":  # to check the pipeline without downloading anything
        full = datasets.FakeData(512, (3, img_size, img_size), 10, transform=transforms.ToTensor())
        test = datasets.FakeData(128, (3, img_size, img_size), 10, transform=transforms.ToTensor(), random_offset=1)
    else:
        raise ValueError(f"unknown dataset: {name}")

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(full), generator=g).tolist()
    n_val = int(len(full) * val_fraction)
    return Subset(full, perm[n_val:]), Subset(full, perm[:n_val]), test


def get_ood_dataset(name: str, root: str = "data", img_size: int = 32):
    """Out-of-distribution dataset (never seen in training), for anomaly detection."""
    tf = _tf(img_size)
    if name == "svhn":
        return datasets.SVHN(root, split="test", download=True, transform=tf)
    if name == "fake":
        return datasets.FakeData(128, (3, img_size, img_size), 10, transform=transforms.ToTensor(), random_offset=2)
    raise ValueError(f"unknown OOD dataset: {name}")
