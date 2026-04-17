# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

from typing import Callable, Optional, Tuple, Union, List, Literal

import numpy as np
import math
import einops
import torch
from torch import Tensor
import torch.nn as nn

from sparsh.tactile_ssl.utils import create_ndgrid


def make_2tuple(x):
    if isinstance(x, tuple) or isinstance(x, list):
        assert len(x) == 2
        return x

    assert isinstance(x, int)
    return (x, x)


def make_tuple(x):
    if isinstance(x, tuple) or isinstance(x, list):
        return x

    assert isinstance(x, int)
    return (x,)


class PatchEmbed(nn.Module):
    """
    2D image to patch embedding: (B,C,H,W) -> (B,N,D)

    Args:
        img_size: Image size.
        patch_size: Patch token size.
        in_chans: Number of input image channels.
        embed_dim: Number of linear projection output channels.
        norm_layer: Normalization layer.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = (
            image_HW[0] // patch_HW[0],
            image_HW[1] // patch_HW[1],
        )

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches: int = int(patch_grid_size[0] * patch_grid_size[1])

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        H, W = x.shape[-2], x.shape[-1]
        patch_H, patch_W = self.patch_size

        assert (
            H % patch_H == 0
        ), f"Input image height {H} is not a multiple of patch height {patch_H}"
        assert (
            W % patch_W == 0
        ), f"Input image width {W} is not a multiple of patch width: {patch_W}"

        x = self.proj(x)  # B C H W
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # B HW C
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)  # B H W C
        return x

    def flops(self) -> float:
        Ho, Wo = self.patches_resolution
        flops = (
            Ho
            * Wo
            * self.embed_dim
            * self.in_chans
            * (self.patch_size[0] * self.patch_size[1])
        )
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

class PatchEmbed3D(PatchEmbed): 
    """
    2D video to patch embedding: (B, T, C, H, W) -> (B, N, D)
    """
    def __init__(self, 
                 num_frames: int,
                 tubelet_size: int = 2,
                 *args, **kwargs,
                 ):
        super().__init__(*args, **kwargs)
        self.tubelet_size = tubelet_size

        self.num_patches = self.num_patches * (num_frames // self.tubelet_size)
        patch_THW = (tubelet_size, self.patch_size[0], self.patch_size[1])
        self.proj = nn.Conv3d(in_channels=self.in_chans, out_channels=self.embed_dim, kernel_size=patch_THW, stride=patch_THW)


class SinusoidalEmbed(nn.Module):
    """
    Add sinusoidal position embeddings to the patch embeddings.
    """

    def __init__(
        self,
        size: Union[int, List[int]],
        stride: Union[int, List[int]],
        embed_dim: int = 768,
        logspace: bool = False,
    ) -> None:
        super().__init__()
        size = make_tuple(size)
        stride = make_tuple(stride)
        assert (
            len(size) < 4
        ), "Sinusoidal position embeddings only support 1D, 2D and 3D grids."
        assert len(size) == len(stride), "size and stride must have the same length"
        patch_grid_size = [s // stride[i] for i, s in enumerate(size)]
        self.embed_dim = embed_dim
        self.patches_resolution = patch_grid_size
        self.num_patches = np.prod(patch_grid_size)

        assert self.embed_dim % 2 == 0, "Embedding dimension must be divisible by 2"
        self.num_bands = self.embed_dim // 2
        self.num_bands = math.ceil(self.embed_dim / (2 * len(size)))
        resolution = 10000
        if logspace:
            frequency_bands = torch.stack(
                [
                    torch.logspace(
                        0.0, -math.log2(resolution / 2.0), self.num_bands + 1, base=2
                    )[:-1]
                    for _ in range(len(size))
                ],
                dim=0,
            )
        else:
            frequency_bands = torch.stack(
                [
                    torch.linspace(0, 1.0, steps=self.num_bands + 1)[:-1]
                    for _ in range(len(size))
                ],
                dim=0,
            )
            frequency_bands = resolution**-frequency_bands
        self.register_buffer("frequency_bands", frequency_bands)
        self.register_buffer("cached_encoding", None, persistent=False)

    def forward(
        self,
        device: torch.device,
        normalized_coords: bool = False,
    ):
        """
        Args:
            pos: Position (normalised Index [-1, 1]) tensor of shape (num_pos, num_dims)
        Returns:
            encoded_pos: Encoded position tensor of shape (num_pos, num_dims, 2*num_bands)
        """
        if self.cached_encoding is not None:
            if self.cached_encoding.device == device:
                return self.cached_encoding
            else:
                return self.cached_encoding.to(device, non_blocking=True)
        grid = create_ndgrid(
            self.patches_resolution, device=device, normalized_coords=normalized_coords
        )
        if grid.dim() < 2:
            grid = grid[..., None]
        freq_bands_buf = self.get_buffer("frequency_bands")
        features = grid[..., None] * freq_bands_buf

        encoded_pos_sin = torch.sin(features)
        encoded_pos_cos = torch.cos(features)

        encoded_pos = torch.cat([encoded_pos_sin, encoded_pos_cos], dim=-1)
        encoded_pos = encoded_pos.flatten(-2, -1)
        self.cached_encoding = encoded_pos[..., : self.embed_dim]
        return self.cached_encoding

    def forward_with_x(self, x: Tensor, normalized_coords: bool = False) -> Tensor:
        freq_bands_buf = self.get_buffer("frequency_bands")
        features = x[..., None] * freq_bands_buf

        encoded_pos_sin = torch.sin(features)
        encoded_pos_cos = torch.cos(features)

        encoded_pos = torch.cat([encoded_pos_sin, encoded_pos_cos], dim=-1)
        encoded_pos = encoded_pos.flatten(-2, -1)
        return encoded_pos[..., : self.embed_dim]


if __name__ == "__main__":
    sin_embed = SinusoidalEmbed([224, 224], [16, 16], 768).to(torch.device("cuda"))
    print(sin_embed(torch.device("cuda")))
