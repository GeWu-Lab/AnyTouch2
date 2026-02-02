# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

import os
from typing import Callable, List, Any, Tuple, Dict
import warnings

import torch
from torch import nn, Tensor

from .attention import Attention, CrossAttention
from .drop_path import DropPath
from .layer_scale import LayerScale
from .mlp import Mlp


XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import fmha, scaled_index_add, index_select_cat

        XFORMERS_AVAILABLE = True
        warnings.warn("xFormers is available (Block)")
    else:
        warnings.warn("xFormers is disabled (Block)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False

    warnings.warn("xFormers is not available (Block)")


class DecoderBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        proj_bias: bool = False,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        self_attn_class: Callable[..., nn.Module] = Attention,
        cross_attn_class: Callable[..., nn.Module] = CrossAttention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.self_attn = self_attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.q_norm2 = norm_layer(dim)
        self.kv_norm2 = norm_layer(dim)
        self.cross_attn = cross_attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls3 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path3 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    def forward(self, q, kv):
        def self_attn_residual_func(q: Tensor) -> Tensor:
            return self.ls1(self.self_attn(self.norm1(q)))

        def cross_attn_residual_func(q: Tensor, kv: Tensor) -> Tensor:
            return self.ls2(self.cross_attn(self.q_norm2(q), self.kv_norm2(kv)))

        def ffn_residual_func(q: Tensor) -> Tensor:
            return self.ls3(self.mlp(self.norm3(q)))

        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            q = drop_add_residual_stochastic_depth(
                [q],
                residual_func=self_attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            q = drop_add_residual_stochastic_depth(
                [q, kv],
                residual_func=cross_attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            q = drop_add_residual_stochastic_depth(
                [q],
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            q = q + self.drop_path1(self_attn_residual_func(q))
            q = q + self.drop_path2(cross_attn_residual_func(q, kv))
            q = q + self.drop_path3(ffn_residual_func(q))
        else:
            q = q + self_attn_residual_func(q)
            q = q + cross_attn_residual_func(q, kv)
            q = q + ffn_residual_func(q)

        return q


def drop_add_residual_stochastic_depth(
    xs: List[Tensor],
    residual_func: Callable[[List[Tensor]], Tensor],
    sample_drop_ratio: float = 0.0,
) -> Tensor:
    # 1) extract subset using permutation
    q = xs[0]
    b, _, _ = q.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=q.device))[:sample_subset_size]
    xs_subset = [x[brange] for x in xs]

    # 2) apply residual_func to get residual
    residual = residual_func(*xs_subset)

    q_flat = q.flatten(1)
    residual = residual.flatten(1)

    residual_scale_factor = b / sample_subset_size

    # 3) add the residual
    q_plus_residual = torch.index_add(
        q_flat, 0, brange, residual.to(dtype=q.dtype), alpha=residual_scale_factor
    )
    return q_plus_residual.view_as(q)


def get_branges_scales(xs: List[Tensor], sample_drop_ratio: float = 0.0):
    q = xs[0]
    b, _, _ = q.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=q.device))[:sample_subset_size]
    residual_scale_factor = b / sample_subset_size
    return brange, residual_scale_factor


def add_residual(
    x: Tensor,
    brange: Tensor,
    residual: Tensor,
    residual_scale_factor: float,
    scaling_vector=None,
):
    if scaling_vector is None:
        x_plus_residual = torch.index_add(
            x.flatten(1),
            0,
            brange,
            residual.flatten(1).to(dtype=x.dtype),
            alpha=residual_scale_factor,
        )
    else:
        x_plus_residual = scaled_index_add(
            x,
            brange,
            residual.to(dtype=x.dtype),
            scaling=scaling_vector,
            alpha=residual_scale_factor,
        )
    return x_plus_residual


attn_bias_cache: Dict[Tuple, Any] = {}
