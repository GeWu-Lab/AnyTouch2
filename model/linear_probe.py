import torch
import numpy as np

from torch import Tensor, nn
from transformers import CLIPModel as HFCLIPModel, CLIPVisionConfig
from timm.models.layers import trunc_normal_
from model.tactile_mae import TactileVideoMAE
from model.layers import CrossAttention, CrossAttentionBlock
from model.layers.block import Block
import math

class AttentivePooler(nn.Module):
    """Attentive Pooler"""

    def __init__(
        self,
        num_queries=1,
        embed_dim=768,
        num_heads=16,
        mlp_ratio=4.0,
        depth=1,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        qkv_bias=True,
        complete_block=True,
    ):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, embed_dim))

        self.complete_block = complete_block
        if complete_block:
            self.cross_attention_block = CrossAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
            )
        else:
            self.cross_attention_block = CrossAttention(
                dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias
            )

        self.blocks = None
        if depth > 1:
            self.blocks = nn.ModuleList(
                [
                    Block(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=False,
                        norm_layer=norm_layer,
                    )
                    for i in range(depth - 1)
                ]
            )

        self.init_std = init_std
        torch.nn.init.trunc_normal_(self.query_tokens, std=self.init_std)
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        if self.complete_block:
            rescale(self.cross_attention_block.xattn.proj.weight.data, 1)
            rescale(self.cross_attention_block.mlp.fc2.weight.data, 1)
        else:
            rescale(self.cross_attention_block.proj.weight.data, 1)
        if self.blocks is not None:
            for layer_id, layer in enumerate(self.blocks, 1):
                rescale(layer.attn.proj.weight.data, layer_id + 1)
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        q = self.query_tokens.repeat(len(x), 1, 1)
        # print(q.shape, x.shape)
        q = self.cross_attention_block(q, x)
        if self.blocks is not None:
            for blk in self.blocks:
                q = blk(q)
        return q

class ForceHead(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio, depth, norm_layer, init_std, qkv_bias, complete_block, classes=3):
        super(ForceHead, self).__init__()

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
        self.probe = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, classes),
        )

    def forward(self, x):
        x = self.pooler(x).squeeze(1)
        x = self.probe(x)
        return x

class AttnHead(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio, depth, norm_layer, init_std, qkv_bias, complete_block, classes=3):
        super(AttnHead, self).__init__()

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
        self.probe = nn.Sequential(
            nn.Linear(embed_dim, classes)
        )

    def forward(self, x):
        x = self.pooler(x).squeeze(1)
        x = self.probe(x)
        return x


class TactileProbeVideo(nn.Module):
    def __init__(self, args, config, num_frames, tube_size):
        super(TactileProbeVideo, self).__init__()

        config.vision_config.num_frames = num_frames
        config.vision_config.tube_size = tube_size # Unused

        self.tactile_model = TactileVideoMAE(args, config, num_frames, tube_size)
        
        self.pooling = args.pooling

        if args.dataset == 'material':
            self.classes = 20
        elif args.dataset == 'cloth':
            self.classes = 20
        elif args.dataset == 'rough':
            self.classes = 1
        elif args.dataset == 'hard':
            self.classes = 1
        elif args.dataset == 'obj2' or args.dataset == 'obj1' or args.dataset == 'objreal':
            self.classes = 7
        else:
            print('sparsh bench')
        
        if 'force' in args.dataset:
            self.head = ForceHead(
                embed_dim=config.vision_config.hidden_size,
                num_heads=16,
                mlp_ratio=4.0,
                depth=1,
                norm_layer=nn.LayerNorm,
                init_std=0.02,
                qkv_bias=True,
                complete_block=True
            )
            self.pooling = 'none'
        elif args.dataset in ['cloth']:
            self.head = AttnHead(
                embed_dim=config.vision_config.hidden_size,
                num_heads=16,
                mlp_ratio=4.0,
                depth=1,
                norm_layer=nn.LayerNorm,
                init_std=0.02,
                qkv_bias=True,
                complete_block=True,
                classes = self.classes
            )
            self.pooling = 'none'
        else:
            if self.pooling == 'cls':
                self.head = nn.Linear(config.projection_dim, self.classes)
            else:
                self.head = nn.Linear(config.vision_config.hidden_size, self.classes)

        self.dataset = args.dataset

        self.single_patch_num = (config.vision_config.image_size // config.vision_config.patch_size) ** 2


    def init_head(self):
        trunc_normal_(self.head.weight, std=0.01)
    
    def forward(self, x, sensor_type = None, return_feature = False):

        with torch.no_grad():
            if self.pooling == 'none':
                x = self.tactile_model(x, sensor_type = sensor_type, probe=True)
                out = x
            else:
                if self.pooling == 'cls':
                    # out = self.touch_projection(x.pooler_output)
                    x = self.tactile_model(x, sensor_type = sensor_type, probe=True, get_cls=True)
                    out = x
                elif self.pooling == 'last':
                    x = self.tactile_model(x, sensor_type = sensor_type, probe=True, get_cls=False)
                    out = x[:, -self.single_patch_num:, :]
                else:
                    x = self.tactile_model(x, sensor_type = sensor_type, probe=True, get_cls=False)
                    if self.use_sensor_token:
                        out = x[:, 6:, :]
                    else:
                        out = x[:, 1:, :]

        feature = out
        # print(out.shape)

        if self.pooling == 'none':
            out = self.head(out)

        else:
            if self.pooling == 'cls':
                out = self.head(out)

            elif self.pooling == 'global' or self.pooling == 'last':
                out = self.head(out.mean(dim=1))

        
        if return_feature:
            return out, feature
        return out
