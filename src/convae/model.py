"""Convolutional autoencoder: image -> latent vector z (dim d) -> image."""
import torch
from torch import nn


def down_block(c_in: int, c_out: int) -> nn.Sequential:
    # Stride-2 conv halves the resolution (learned downsampling, no max-pool).
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


def up_block(c_in: int, c_out: int) -> nn.Sequential:
    # Upsample + Conv instead of ConvTranspose2d: avoids checkerboard artifacts.
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class Encoder(nn.Module):
    def __init__(self, latent_dim: int, channels=(64, 128, 256), in_ch: int = 3, img_size: int = 32):
        super().__init__()
        layers, c = [], in_ch
        for h in channels:
            layers.append(down_block(c, h))
            c = h
        self.features = nn.Sequential(*layers)
        self.spatial = img_size // 2 ** len(channels)  # 32 -> 4
        self.fc = nn.Linear(c * self.spatial**2, latent_dim)  # bottleneck: vector of size d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.features(x).flatten(1))


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, channels=(64, 128, 256), out_ch: int = 3, img_size: int = 32):
        super().__init__()
        rev = list(channels)[::-1]  # [256, 128, 64]
        self.c0 = rev[0]
        self.spatial = img_size // 2 ** len(channels)
        self.fc = nn.Linear(latent_dim, self.c0 * self.spatial**2)
        outs = rev[1:] + [rev[-1] // 2]  # [128, 64, 32]
        self.ups = nn.Sequential(*[up_block(i, o) for i, o in zip(rev, outs)])
        self.head = nn.Conv2d(outs[-1], out_ch, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1, self.c0, self.spatial, self.spatial)
        # Sigmoid: output in [0, 1], same range as the input images.
        return torch.sigmoid(self.head(self.ups(h)))


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 64, channels=(64, 128, 256), img_size: int = 32):
        super().__init__()
        self.encoder = Encoder(latent_dim, channels, img_size=img_size)
        self.decoder = Decoder(latent_dim, channels, img_size=img_size)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        return self.decode(z), z
