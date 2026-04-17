# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import math
from functools import partial
from typing import Callable, Literal, Sequence, Tuple, Union

import einops
import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_

from sparsh.tactile_ssl.utils import apply_masks

from .layers import MemEffAttention, Mlp
from .layers import NestedTensorBlock as Block
from .layers import PatchEmbed, PatchEmbed3D, SinusoidalEmbed, SwiGLUFFNFused

logger = logging.getLogger(__name__)


def named_apply(
    fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False
) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(
            fn=fn,
            module=child_module,
            name=child_name,
            depth_first=depth_first,
            include_root=True,
        )
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class BlockChunk(nn.ModuleList):
    def forward(self, x):
        for b in self:
            x = b(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = (224, 224),
        patch_size: int = 16,
        num_frames: int = 1,
        tubelet_size: int = 2,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        ffn_bias: bool = True,
        proj_bias: bool = True,
        drop_path_rate: float = 0.0,
        drop_path_uniform: bool = False,
        init_values: int = 1,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed,
        pos_embed_fn: Literal["sinusoidal", "learned"] = "learned",
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks: int = 0,
        num_register_tokens: int = 0,
        interpolate_antialias: bool = False,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            drop_path_rate (float): stochastic depth rate
            drop_path_uniform (bool): apply uniform drop rate across blocks
            weight_init (str): weight init scheme
            init_values (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            act_layer (nn.Module): MLP activation layer
            block_fn (nn.Module): transformer block class
            ffn_layer (str): "mlp", "swiglu", "swiglufused" or "identity"
            block_chunks: (int) split block sequence into block_chunks units for FSDP wrap
            num_register_tokens: (int) number of extra cls tokens (so-called "registers")
            interpolate_antialias: (str) flag to apply anti-aliasing when interpolating positional embeddings
        """
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        img_size = (
            (img_size, img_size) if isinstance(img_size, int) else tuple(img_size)
        )
        assert len(img_size) == 2, "Vision Transformer only works with 2D images"

        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )
        self.n_blocks = depth

        # Video params
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.is_video = self.num_frames > 1

        self.num_heads = num_heads
        self.img_size = img_size
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.pos_embed_fn = pos_embed_fn

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        if self.is_video:
            self.patch_embed = PatchEmbed3D(
                num_frames,
                tubelet_size=self.tubelet_size,
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
        num_patches = self.patch_embed.num_patches

        assert num_register_tokens >= 0
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
            if num_register_tokens
            else None
        )
        if pos_embed_fn == "sinusoidal":
            if self.is_video:
                self.pos_embed = SinusoidalEmbed(
                    [num_frames] + list(self.img_size),
                    [
                        self.num_frames // self.tubelet_size,
                        self.patch_size,
                        self.patch_size,
                    ],
                    embed_dim=self.embed_dim,
                )
            else:
                self.pos_embed = SinusoidalEmbed(
                    list(self.img_size),
                    [self.patch_size, self.patch_size],
                    embed_dim=self.embed_dim,
                )
        elif (
            pos_embed_fn == "learned"
        ):  # NOTE: Different from DINOv2, we don't add learned positional embedding to cls / register tokens
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [
                x.item() for x in torch.linspace(0, drop_path_rate, depth)
            ]  # stochastic depth decay rule

        if ffn_layer == "mlp":
            logger.info("using MLP layer as FFN")
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            logger.info("using SwiGLU layer as FFN")
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            logger.info("using Identity layer as FFN")

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            )
            for i in range(depth)
        ]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                # this is to keep the block index consistent if we chunk the block list
                chunked_blocks.append(
                    [nn.Identity()] * i + blocks_list[i : i + chunksize]
                )
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.init_weights()
        self._rescale_blocks()

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def init_weights(self):
        if self.pos_embed_fn == "learned":
            trunc_normal_(self.pos_embed, std=0.02)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)

        named_apply(init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, img_shape, img_dtype, device):
        previous_dtype = img_dtype
        pos_embed = None
        if self.pos_embed_fn == "sinusoidal":
            pos_embed = self.pos_embed(device).float().unsqueeze(0)
        elif self.pos_embed_fn == "learned":
            pos_embed = self.pos_embed.float()
        else:
            raise NotImplementedError("Unknown position embedding function")

        if self.is_video:
            _, _, t, h, w = img_shape
            if h == self.img_size[0] and w == self.img_size[1] and t == self.num_frames:
                return pos_embed

            dim = pos_embed.shape[-1]
            t0 = t // self.tubelet_size
            w0 = w // self.patch_size
            h0 = h // self.patch_size

            pos_embed = nn.functional.interpolate(
                pos_embed.reshape(1, t0, w0, h0, dim).permute(0, 4, 1, 2, 3),
                mode="trilinear",
                antialias=self.interpolate_antialias,
                size=(t0, w0, h0),
            )
            assert (t0, w0, h0) == pos_embed.shape[-3:]
            pos_embed = pos_embed.permute(0, 2, 3, 4, 1).view(1, -1, dim)
            return pos_embed

        _, _, h, w = img_shape
        if h == self.img_size[0] and w == self.img_size[1]:
            return pos_embed

        dim = pos_embed.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size

        pos_embed = nn.functional.interpolate(
            pos_embed.reshape(1, w0, h0, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            size=(w0, h0),
        )
        assert (w0, h0) == pos_embed.shape[-2:]
        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return pos_embed.to(previous_dtype)

    def prepare_tokens_with_masks(self, x, masks=None):

        pos_encoding = self.interpolate_pos_encoding(x.shape, x.dtype, device=x.device)
        x = self.patch_embed(x)
        x = x + pos_encoding

        if masks is not None:
            x = apply_masks(x, masks)

        if self.register_tokens is not None:
            x = torch.cat(
                (
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x,
                ),
                dim=1,
            )
        return x

    def forward_features_list(self, x_list, masks_list):
        x = [
            self.prepare_tokens_with_masks(x, masks)
            for x, masks in zip(x_list, masks_list)
        ]
        for blk in self.blocks:
            x = blk(x)

        all_x = x
        output = []
        for x, masks in zip(all_x, masks_list):
            x_norm = self.norm(x)
            output.append(
                {
                    "x_norm_regtokens": x_norm[:, : self.num_register_tokens],
                    "x_norm_patchtokens": x_norm[:, self.num_register_tokens :],
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward_features(self, x, masks=None):
        if isinstance(x, list):
            return self.forward_features_list(x, masks)

        x = self.prepare_tokens_with_masks(x, masks)

        for blk in self.blocks:
            x = blk(x)

        x_norm = self.norm(x)
        return {
            "x_norm_regtokens": x_norm[:, : self.num_register_tokens],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens :],
            "x_prenorm": x,
            "masks": masks,
        }

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = (
            range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        )
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(
            blocks_to_take
        ), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def _get_intermediate_layers_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        output, i, total_block_len = [], 0, len(self.blocks[-1])
        # If n is an int, take the n last blocks. If it's a list, take them
        blocks_to_take = (
            range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        )
        for block_chunk in self.blocks:
            for blk in block_chunk[i:]:  # Passing the nn.Identity()
                x = blk(x)
                if i in blocks_to_take:
                    output.append(x)
                i += 1
        assert len(output) == len(
            blocks_to_take
        ), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        if self.chunked_blocks:
            outputs = self._get_intermediate_layers_chunked(x, n)
        else:
            outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1 + self.num_register_tokens :] for out in outputs]
        if reshape:
            B, _, w, h = x.shape
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def forward(self, *args, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        return ret["x_norm_patchtokens"]


class VisionTransformerPredictor(VisionTransformer):
    def __init__(
        self,
        input_dim: int,
        num_mask_tokens: int = 1,
        zero_init_mask_tokens: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.input_dim = input_dim
        self.num_mask_tokens = num_mask_tokens
        self.input_projection = nn.Linear(self.input_dim, self.embed_dim)
        self.output_projection = nn.Linear(self.embed_dim, input_dim)
        self.patch_embed = nn.Identity()
        self.mask_token = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.embed_dim))] * num_mask_tokens
        )
        if not zero_init_mask_tokens:
            for mt in self.mask_token:
                trunc_normal_(mt, std=0.02)
        self.init_weights()

    def forward(self, x, context_masks, masks=None, mask_index=1):
        b, num_context_tokens, c = x.shape
        # Batch size for x is b * len(context_masks)
        batch_size = b // len(context_masks)

        x = self.input_projection(x)

        assert (
            self.pos_embed_fn == "sinusoidal"
        ), "Only sinusoidal positional encoding is supported for Predictor."

        pos_emb_x = self.pos_embed(x.device)
        pos_emb_x = einops.repeat(pos_emb_x, "n c -> b n c", b=batch_size)
        x = x + apply_masks(pos_emb_x, context_masks)

        pos_emb_masked = apply_masks(pos_emb_x, masks)
        pos_emb_masked = einops.repeat(
            pos_emb_masked,
            "(k b) n c -> (p k b) n c",
            k=len(masks),
            p=len(context_masks),
        )

        mask_token = self.mask_token[mask_index % self.num_mask_tokens]

        prediction_tokens = einops.repeat(
            mask_token,
            "1 c -> b n c",
            b=pos_emb_masked.shape[0],
            n=pos_emb_masked.shape[1],
        )
        prediction_tokens = prediction_tokens + pos_emb_masked
        x = einops.repeat(
            x, "(p b) n c -> (p k b) n c", p=len(context_masks), k=len(masks)
        )
        x = torch.cat([x, prediction_tokens], dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x[:, num_context_tokens:]

        x = self.output_projection(x)
        return x


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """vit weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


def vit_predictor(
    input_dim,
    patch_size=16,
    num_register_tokens=0,
    embed_dim=384,
    depth=6,
    num_heads=12,
    **kwargs,
):
    model = VisionTransformerPredictor(
        input_dim=input_dim,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_tiny(patch_size=16, depth=12, num_register_tokens=0, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=192,
        # depth=12,
        depth=depth,
        num_heads=3,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_small(patch_size=16, num_register_tokens=0, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_base(patch_size=16, num_register_tokens=0, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_large(patch_size=16, num_register_tokens=0, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_giant2(patch_size=16, num_register_tokens=0, **kwargs):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
    """
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model

VIT_EMBED_DIMS = {
    'vit_tiny': 192,
    'vit_small': 384,
    'vit_base': 768,
    'vit_large': 1024,
    'vit_giant2': 1536,
    'anytouch2_base': 512,
}