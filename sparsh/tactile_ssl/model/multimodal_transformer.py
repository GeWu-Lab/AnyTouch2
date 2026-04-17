# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import math
from functools import partial
from typing import Callable, Optional, List, Literal
import numpy as np

import einops
import torch
import torch.nn as nn

from sparsh.tactile_ssl.model.vision_transformer import init_weights_vit_timm
from sparsh.tactile_ssl.utils.logging import get_pylogger

from .layers import MemEffAttention, Mlp, MemEffCrossAttention
from .layers import NestedTensorBlock as Block
from .layers import DecoderBlock
from .layers import SinusoidalEmbed, SwiGLUFFNFused
from abc import abstractmethod

log = get_pylogger(__name__)


class MultimodalTransformer(nn.Module):
    def __init__(
        self,
        modal_dims: List[int],
        modal_lens: List[int],
        embed_dim: int,
        depth: int = 12,
        block_class: Callable[..., nn.Module] = partial(
            Block, attn_class=MemEffAttention
        ),
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        ffn_layer: str = "mlp",
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        pos_embed_fn: Literal["sinusoidal", "learned"] = "learned",
        init_values: Optional[float] = None,
        num_register_tokens: int = 0,
        drop_path_rate: float = 0.0,
        drop_path_uniform: bool = False,
        shared_attn: bool = True,
    ):
        assert len(modal_dims) == len(modal_lens)
        super().__init__()
        self.modal_dims = modal_dims
        self.modal_lens = modal_lens
        self.modal_sizes = [d * l for d, l in zip(self.modal_dims, self.modal_lens)]
        self.embed_dim = embed_dim
        self.num_dims = len(modal_dims)
        self.depth = depth
        self.num_heads = num_heads
        self.shared_attn = shared_attn

        assert num_register_tokens >= 0
        self.num_register_tokens = num_register_tokens
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
            if num_register_tokens > 0
            else None
        )

        self.init_pos_embed(pos_embed_fn)

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [
                x.item() for x in torch.linspace(0, drop_path_rate, depth)
            ]  # stochastic depth decay rule

        ffn_layer_ = None
        if ffn_layer == "mlp":
            log.info("using MLP layer as FFN")
            ffn_layer_ = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            log.info("using SwiGLU layer as FFN")
            ffn_layer_ = SwiGLUFFNFused
        elif ffn_layer == "identity":
            log.info("using Identity layer as FFN")

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer_ = f
        else:
            raise NotImplementedError

        block_size = 1 if self.shared_attn else self.num_dims
        blocks_list = [
            nn.ModuleList(
                [
                    block_class(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        proj_bias=proj_bias,
                        ffn_bias=ffn_bias,
                        drop_path=dpr[i],
                        norm_layer=norm_layer,
                        act_layer=act_layer,
                        ffn_layer=ffn_layer_,
                        init_values=init_values,
                    )
                    for _ in range(block_size)
                ]
            )
            for i in range(depth)
        ]

        self.blocks = nn.ModuleList(blocks_list)
        self.norm = norm_layer(embed_dim)

        self.init_weights()
        self._rescale_blocks()

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layers in enumerate(self.blocks):
            for layer in layers:
                if layer is Block:
                    rescale(layer.attn.proj.weight.data, layer_id + 1)
                    rescale(layer.mlp.fc2.weight.data, layer_id + 1)
                elif layer is DecoderBlock:
                    rescale(layer.self_attn.proj.weight.data, layer_id + 1)
                    rescale(layer.cross_attn.proj.weight.data, layer_id + 1)
                    rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def init_pos_embed(self, pos_embed_fn):
        self.pos_embed_fn = pos_embed_fn
        if pos_embed_fn == "sinusoidal":
            self.pos_embed = SinusoidalEmbed(
                [sum(self.modal_sizes)],
                [1],
                embed_dim=self.embed_dim,
            )
        elif (
            pos_embed_fn == "learned"
        ):  # NOTE: Different from DINOv2, we don't add learned positional embedding to cls / register tokens
            self.pos_embed = nn.Parameter(
                torch.zeros(
                    1,
                    sum(self.modal_sizes),
                    self.embed_dim,
                )
            )

    def init_weights(self):
        if self.pos_embed_fn == "learned":
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.register_tokens is not None:
            nn.init.trunc_normal_(self.register_tokens, std=1e-6)
        self.apply(init_weights_vit_timm)

    def apply_mask(
        self, x: torch.Tensor, mask: torch.Tensor, mask_type: Literal["block", "tube"]
    ):
        if mask_type == "tube":
            return self.apply_tubelet_mask(x, mask)
        elif mask_type == "block":
            return self.apply_block_mask(x, mask)
        else:
            raise NotImplementedError("We only support tube and block masking.")

    def apply_tubelet_mask(self, x: torch.Tensor, mask: torch.Tensor):
        _, c, t, _ = x.shape
        mask_keep = einops.repeat(mask, "b n -> b c t n", c=c, t=t)
        return torch.gather(x, dim=-1, index=mask_keep)

    def apply_block_mask(self, x: torch.Tensor, mask: torch.Tensor):
        _, c, _, n = x.shape
        mask_keep = einops.repeat(mask, "b t -> b c t n", c=c, n=n)
        return torch.gather(x, dim=-2, index=mask_keep)

    @abstractmethod
    def pre_embed(self, xs: List[torch.Tensor], *args, **kwargs):
        raise NotImplementedError

    def embed(self, xs: List[torch.Tensor]):
        if self.pos_embed_fn == "sinusoidal":
            pos_embed = self.pos_embed(xs[0].device).float().unsqueeze(0)
        elif self.pos_embed_fn == "learned":
            pos_embed = self.pos_embed.float()
        else:
            raise NotImplementedError("Unknown position embedding function")
        pos_embeds = pos_embed.split(self.modal_sizes, dim=-2)
        pos_embeds = [
            einops.rearrange(pos_embed, "b (n t) c->b c t n", t=l, n=d)
            for pos_embed, l, d in zip(pos_embeds, self.modal_lens, self.modal_dims)
        ]
        xs = [x + pos_embed for x, pos_embed in zip(xs, pos_embeds)]
        return xs

    def prepare_tokens(
        self,
        xs: List[torch.Tensor],
        masks: Optional[List[torch.Tensor]] = None,
        mask_types: Optional[Literal["tube", "block"]] = None,
    ):
        if masks is not None and mask_types is not None:
            assert len(masks) == len(self.modal_dims)
            assert len(mask_types) == len(self.modal_dims)
            xs_masked = [
                self.apply_mask(x, mask, mask_type)
                for x, mask, mask_type in zip(xs, masks, mask_types)
            ]
        else:
            xs_masked = xs

        x = torch.cat(
            [einops.rearrange(x, "b c t n-> b (n t) c") for x in xs_masked], dim=1
        )
        if self.register_tokens is not None:
            x = torch.cat([self.register_tokens.expand(x.shape[0], -1, -1), x], dim=1)

        return x

    def transcode(self, x: torch.Tensor):
        if self.shared_attn:
            for blks in self.blocks:
                x = blks[0](x)
        else:
            q_sizes = tuple(self.modal_sizes)
            qs = x.split(q_sizes, dim=-2)
            for blks in self.blocks:
                qs = [blk(q, x) for q, blk in zip(qs, blks)]
                x = torch.cat(qs, dim=-2)
        x_norm = self.norm(x)
        return x, x_norm

    def post_transcode(
        self,
        x: torch.Tensor,
        x_norm: torch.Tensor,
        masks: Optional[List[torch.Tensor]] = None,
        mask_types: Optional[Literal["tube", "block"]] = None,
        restore_shapes: Optional[List[int]] = None,
    ):
        modal_dims = self.modal_dims
        modal_lens = self.modal_lens
        if masks is not None and mask_types is not None:
            assert len(masks) == len(self.modal_dims)
            assert len(mask_types) == len(self.modal_dims)
            split_sizes = tuple(
                [
                    (l * mask.shape[1] if mask_type == "tube" else d * mask.shape[1])
                    for mask, mask_type, d, l in zip(
                        masks, mask_types, modal_dims, modal_lens
                    )
                ]
            )
        elif restore_shapes is None:
            split_sizes = tuple(self.modal_sizes)
        else:
            assert len(restore_shapes) == len(self.modal_dims)
            split_sizes = tuple(np.prod(modal_shape) for modal_shape in restore_shapes)

        def restore_tokens(
            tokens: torch.Tensor,
        ):
            tokens = torch.split(tokens, split_sizes, dim=-2)
            if masks is not None and mask_types is not None:
                return [
                    (
                        einops.rearrange(token, "b (n t) c -> b c t n", t=l)
                        if mask_type == "tube"
                        else einops.rearrange(token, "b (n t) c -> b c t n", n=d)
                    )
                    for token, mask_type, d, l in zip(
                        tokens, mask_types, modal_dims, modal_lens
                    )
                ]
            elif restore_shapes is None:
                return [
                    einops.rearrange(token, "b (n t) c -> b c t n", t=l, n=d)
                    for token, d, l in zip(tokens, modal_dims, modal_lens)
                ]
            else:
                return [
                    einops.rearrange(
                        token, "b (n t) c -> b c t n", t=shape[0], n=shape[1]
                    )
                    for token, shape in zip(tokens, restore_shapes)
                ]

        reg_tokens = x_norm[:, : self.num_register_tokens]
        patch_tokens = restore_tokens(x_norm[:, self.num_register_tokens :])
        patch_tokens_prenorm = restore_tokens(x[:, self.num_register_tokens :])

        out = {
            "x_norm_regtokens": reg_tokens,
            "x_norm_patchtokens": patch_tokens,
            "x_prenorm": patch_tokens_prenorm,
        }
        return out

    def forward_features(
        self,
        xs: List[torch.Tensor],
        masks: Optional[List[torch.Tensor]] = None,
        mask_types: Optional[Literal["tube", "block"]] = None,
    ):
        assert (not masks and not mask_types) or (masks and mask_types)

        xs = self.pre_embed(xs)
        xs = self.embed(xs)
        x = self.prepare_tokens(xs, masks, mask_types)
        x, x_norm = self.transcode(x)
        outputs = self.post_transcode(x, x_norm, masks, mask_types)
        return outputs

    def forward(self, xs, masks=None, mask_types=None):
        outputs = self.forward_features(xs, masks, mask_types)
        return outputs["x_norm_patchtokens"]


class MultimodalMAEDecoder(MultimodalTransformer):
    def __init__(
        self,
        modal_chans: int,
        shared_attn: bool = True,
        shared_mask_token: bool = False,
        *args,
        **kwargs,
    ):
        block_class = (
            partial(Block, attn_class=MemEffAttention)
            if shared_attn
            else partial(
                DecoderBlock,
                self_attn_class=MemEffAttention,
                cross_attn_class=MemEffCrossAttention,
            )
        )

        super().__init__(
            *args,
            block_class=block_class,
            shared_attn=shared_attn,
            **kwargs,
        )

        self.modal_chans = modal_chans
        self.input_projection = nn.Linear(self.modal_chans, self.embed_dim)
        self.shared_mask_token = shared_mask_token
        self.mask_tokens = (
            nn.Parameter(torch.zeros(1, 1, 1, self.embed_dim))
            if self.shared_mask_token
            else nn.Parameter(torch.zeros(len(self.modal_dims), 1, 1, self.embed_dim))
        )
        nn.init.trunc_normal_(self.mask_tokens, std=0.02)
        super().init_weights()

    def pre_embed(
        self,
        xs: List[torch.Tensor],
        ids_restore: List[torch.Tensor],
        mask_types: List[Literal["tube", "block"]],
    ):
        b = xs[0].shape[0]
        lens_keep = [
            x.shape[-1] if mask_type == "tube" else x.shape[-2]
            for x, mask_type in zip(xs, mask_types)
        ]
        mask_tokens = [
            (
                einops.repeat(
                    mask_token,
                    "1 1 c -> b n t c",
                    b=b,
                    n=d - len_keep,
                    t=l,
                )
                if mask_type == "tube"
                else einops.repeat(
                    mask_token,
                    "1 1 c -> b n t c",
                    b=b,
                    n=d,
                    t=l - len_keep,
                )
            )
            for mask_type, len_keep, l, d, mask_token in zip(
                mask_types,
                lens_keep,
                self.modal_lens,
                self.modal_dims,
                self.mask_tokens.expand(len(self.modal_dims), 1, 1, self.embed_dim),
            )
        ]

        ids_restore = [
            (
                einops.repeat(id_restore, "b n->b n t c", n=d, t=l, c=self.embed_dim)
                if mask_type == "tube"
                else einops.repeat(
                    id_restore, "b t->b n t c", n=d, t=l, c=self.embed_dim
                )
            )
            for id_restore, mask_type, l, d in zip(
                ids_restore, mask_types, self.modal_lens, self.modal_dims
            )
        ]

        xs = [einops.rearrange(x, "b c t n->b n t c") for x in xs]
        xs = [self.input_projection(x) for x in xs]
        xs = [
            (
                torch.gather(
                    torch.cat([x, mask_token], dim=1), index=id_restore, dim=-3
                )
                if mask_type == "tube"
                else torch.gather(
                    torch.cat([x, mask_token], dim=2), index=id_restore, dim=-2
                )
            )
            for x, id_restore, mask_type, mask_token in zip(
                xs, ids_restore, mask_types, mask_tokens
            )
        ]
        xs = [einops.rearrange(x, "b n t c->b c t n") for x in xs]
        return xs

    @abstractmethod
    def post_transcode(self, x: torch.Tensor):
        raise NotImplementedError

    def forward(
        self,
        xs: List[torch.Tensor],
        ids_restore: List[torch.Tensor],
        mask_types: List[Literal["tube", "block"]],
    ):
        assert len(xs) == len(self.modal_dims)
        assert len(ids_restore) == len(self.modal_dims)
        assert len(mask_types) == len(self.modal_dims)

        xs = self.pre_embed(xs, ids_restore, mask_types)
        xs = self.embed(xs)
        x = self.prepare_tokens(xs)
        _, x_norm = self.transcode(x)
        output = self.post_transcode(x_norm)
        return output
