# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from sparsh.tactile_ssl.downstream_task.attentive_pooler import AttentivePooler
from sparsh.tactile_ssl.model import VIT_EMBED_DIMS

class SlipProbe(nn.Module):
    def __init__(
        self,
        attn_pool=False,
        embed_dim=768,
        num_heads=12,
        mlp_ratio=4.0,
        depth=1,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        qkv_bias=True,
        complete_block=True,
        num_classes=2,
        dropout=0.0,
        with_force_input=False,
        is_anytouch=False,
    ):
        super().__init__()
        self.attn_pool = attn_pool
        self.with_force_input = with_force_input
        if attn_pool:
            self.pooler = AttentivePooler(
                num_queries=1,
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                depth=depth,
                norm_layer=norm_layer,
                init_std=init_std,
                qkv_bias=qkv_bias,
                complete_block=complete_block,
            )

        in_features = embed_dim if not with_force_input else embed_dim + 3
        self.fc_norm = norm_layer(in_features)
        self.probe_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.probe = nn.Sequential(
            nn.Linear(in_features, in_features // 4),
            nn.Sigmoid(),
            nn.Linear(in_features // 4, num_classes),
        )

    def forward(self, inputs):
        x = inputs["latent"]
        if self.attn_pool:
            x = self.pooler(x).squeeze(1)
        else:
            x = x.mean(dim=1)
        if self.with_force_input:
            x = torch.cat([x, inputs["force"]], dim=1)

        x = self.fc_norm(x)
        x = self.probe_dropout(x)
        x = self.probe(x)
        output = {"slip": x}
        return output


class SlipForceProbe(nn.Module):
    def __init__(
        self,
        attn_pool=False,
        embed_dim='base',
        num_heads=12,
        mlp_ratio=4.0,
        depth=1,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        qkv_bias=True,
        complete_block=True,
        num_classes=2,
        dropout=0.0,
        is_anytouch=False,
    ):
        super().__init__()
        embed_dim = VIT_EMBED_DIMS[f"vit_{embed_dim}"]
        self.attn_pool = attn_pool
        if attn_pool:
            self.pooler = AttentivePooler(
                num_queries=1,
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                depth=depth,
                norm_layer=norm_layer,
                init_std=init_std,
                qkv_bias=qkv_bias,
                complete_block=complete_block,
            )
        # force decoder
        self.force_probe = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, 3),
        )
        # slip decoder
        self.fc_norm = norm_layer(embed_dim)
        self.probe_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.probe = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.Sigmoid(),
            nn.Linear(embed_dim // 4, num_classes),
        )

    def forward(self, inputs):
        x = inputs["latent"]
        if self.attn_pool:
            x = self.pooler(x).squeeze(1)
        else:
            x = x.mean(dim=1)
        x = self.fc_norm(x)
        x = self.probe_dropout(x)

        # slip decoder
        slip = self.probe(x)

        # force decoder
        force = self.force_probe(x)
        force = F.hardtanh(force)

        output = {"slip": slip, "force": force}
        return output
