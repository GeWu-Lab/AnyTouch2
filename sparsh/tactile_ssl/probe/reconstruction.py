# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple
import math
from functools import partial

import torch
import torch.nn as nn

from sparsh.tactile_ssl.model.vision_transformer import VisionTransformer


class DecoderViT(VisionTransformer):
    def __init__(
        self,
        input_embed_dim: int=768,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.patch_embed = nn.Identity()
        self.decoder_embed = nn.Linear(input_embed_dim, self.embed_dim, bias=True)
        output_dim = self.patch_size * self.patch_size * self.in_chans
        self.decoder_pred = nn.Linear(self.embed_dim, output_dim, bias=True)

        self.init_weights()

    def forward(self, x, img_shape):
        pos_embed = self.interpolate_pos_encoding(img_shape, img_dtype=x.dtype, device=x.device)
        x = self.decoder_embed(x)
        x = x + pos_embed 
        for blk in self.blocks:
            x = blk(x)
        x_norm = self.norm(x)
        x = self.decoder_pred(x_norm)
        return x
    
class MaskDecoderViT(VisionTransformer):
    def __init__(
        self, 
        input_embed_dim: int = 768,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.patch_embed = nn.Identity()
        self.decoder_embed = nn.Linear(input_embed_dim, self.embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        torch.nn.init.normal_(self.mask_token, std=0.02)

        output_dim = self.patch_size * self.patch_size * self.in_chans
        self.decoder_pred = nn.Linear(self.embed_dim, output_dim, bias=True)
        self.init_weights()

    def forward(self, x, img_shape, ids_restore):

        pos_embed = self.interpolate_pos_encoding(img_shape, img_dtype=x.dtype, device=x.device)
        x = self.decoder_embed(x)

        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 0:, :], mask_tokens], dim=1)
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )

        x = x_ + pos_embed
        for blk in self.blocks:
            x = blk(x)
        x_norm = self.norm(x)
        x = self.decoder_pred(x_norm)
        return x

def DecoderImage(**kwargs):
    model = DecoderViT(
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        pos_embed_fn='sinusoidal',
        **kwargs
    )
    return model
