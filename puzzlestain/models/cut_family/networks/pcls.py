"""PCLS (Prototype-Consistent Learning Strategy) segmentation network for PSPStain.

Original source: https://github.com/ccitachi/PSPStain (models/PCLS.py)
Licensed under the project license; see ATTRIBUTION.md.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two convolution layers with batch norm and leaky relu."""

    def __init__(self, in_channels: int, out_channels: int, dropout_p: float) -> None:
        super().__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_conv(x)


class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock."""

    def __init__(self, in_channels: int, out_channels: int, dropout_p: float) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upsampling followed by ConvBlock."""

    def __init__(
        self,
        in_channels1: int,
        in_channels2: int,
        out_channels: int,
        dropout_p: float,
        mode_upsampling: int = 1,
    ) -> None:
        super().__init__()
        self.mode_upsampling = mode_upsampling
        if mode_upsampling == 0:
            self.up = nn.ConvTranspose2d(
                in_channels1, in_channels2, kernel_size=2, stride=2
            )
        elif mode_upsampling == 1:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        elif mode_upsampling == 2:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode="nearest")
        elif mode_upsampling == 3:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode="bicubic", align_corners=True)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if self.mode_upsampling != 0:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(self, params: dict) -> None:
        super().__init__()
        self.params = params
        self.in_chns = self.params["in_chns"]
        self.ft_chns = self.params["feature_chns"]
        self.n_class = self.params["class_num"]
        self.dropout = self.params["dropout"]
        assert len(self.ft_chns) == 5
        self.in_conv = ConvBlock(self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]


def masked_average_pooling(feature: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = F.interpolate(
        mask, size=feature.shape[-2:], mode="bilinear", align_corners=True
    )
    masked_feature = torch.sum(feature * mask, dim=(2, 3)) / (
        mask.sum(dim=(2, 3)) + 1e-5
    )
    return masked_feature


def batch_prototype(feature: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    batch_pro = torch.zeros(
        mask.shape[0], mask.shape[1], feature.shape[1], device=feature.device
    )
    for i in range(mask.shape[1]):
        classmask = mask[:, i, :, :]
        proclass = masked_average_pooling(feature, classmask.unsqueeze(1))
        batch_pro[:, i, :] = proclass
    return batch_pro


def similarity_calulation(feature: torch.Tensor, batchpro: torch.Tensor) -> torch.Tensor:
    B = feature.size(0)
    feature = feature.view(feature.size(0), feature.size(1), -1)
    feature = feature.transpose(1, 2)
    feature = feature.contiguous().view(-1, feature.size(2))
    C = batchpro.size(1)
    batchpro = batchpro.contiguous().view(-1, batchpro.size(2))
    feature = F.normalize(feature, p=2.0, dim=1)
    batchpro = F.normalize(batchpro, p=2.0, dim=1)
    similarity = torch.mm(feature, batchpro.T)
    similarity = similarity.reshape(-1, B, C)
    similarity = similarity.reshape(B, -1, B, C)
    return similarity


def othersimilaritygen(similarity: torch.Tensor) -> torch.Tensor:
    similarity_ = similarity.clone()
    similarity_ = torch.exp(similarity_)
    similarity__ = torch.zeros(
        2, similarity.size(1), similarity.size(3), device=similarity.device
    )
    for i in range(similarity.shape[2]):
        similarity__[i, :, :] = similarity_[i, :, 1 - i, :]
    similaritysum = similarity__
    similaritysum_union = torch.sum(similaritysum, dim=2).unsqueeze(-1)
    othersimilarity = similaritysum / (similaritysum_union + 1e-8)
    return othersimilarity


class DecoderPro(nn.Module):
    def __init__(self, params: dict) -> None:
        super().__init__()
        self.params = params
        self.in_chns = self.params["in_chns"]
        self.ft_chns = self.params["feature_chns"]
        self.n_class = self.params["class_num"]
        self.up_type = self.params["up_type"]
        assert len(self.ft_chns) == 5

        self.up1 = UpBlock(
            self.ft_chns[4],
            self.ft_chns[3],
            self.ft_chns[3],
            dropout_p=0.0,
            mode_upsampling=self.up_type,
        )
        self.up2 = UpBlock(
            self.ft_chns[3],
            self.ft_chns[2],
            self.ft_chns[2],
            dropout_p=0.0,
            mode_upsampling=self.up_type,
        )
        self.up3 = UpBlock(
            self.ft_chns[2],
            self.ft_chns[1],
            self.ft_chns[1],
            dropout_p=0.0,
            mode_upsampling=self.up_type,
        )
        self.up4 = UpBlock(
            self.ft_chns[1],
            self.ft_chns[0],
            self.ft_chns[0],
            dropout_p=0.0,
            mode_upsampling=self.up_type,
        )

        self.out_conv = nn.Conv2d(
            self.ft_chns[0], self.n_class, kernel_size=3, padding=1
        )

    def forward(self, feature: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        x0, x1, x2, x3, x4 = feature

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)

        output = self.out_conv(x)
        mask = torch.softmax(output, dim=1)
        batch_pro = batch_prototype(x, mask)
        similarity_map = similarity_calulation(x, batch_pro)
        other_simi_map = othersimilaritygen(similarity_map)
        return output, other_simi_map


class UNetPro(nn.Module):
    """Lightweight UNet used by PCLS to produce segmentation logits and
    cross-image prototype similarity maps.
    """

    def __init__(self, in_chns: int = 3, class_num: int = 2) -> None:
        super().__init__()
        params1 = {
            "in_chns": in_chns,
            "feature_chns": [32, 32, 64, 128, 256],
            "dropout": [0.05, 0.1, 0.2, 0.3, 0.5],
            "class_num": class_num,
            "up_type": 1,
            "acti_func": "relu",
        }
        self.encoder = Encoder(params1)
        self.decoder1 = DecoderPro(params1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature = self.encoder(x)
        output, other_simi_map = self.decoder1(feature)
        return output, other_simi_map


__all__ = ["UNetPro", "UNet_pro"]

# Alias matching the original reference name.
UNet_pro = UNetPro
