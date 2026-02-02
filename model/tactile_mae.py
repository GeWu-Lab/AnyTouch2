import sys

import copy

import torch
import numpy as np
from einops import rearrange
from typing import Optional, Tuple, Union

from torch import nn
from transformers import CLIPModel as HFCLIPModel, CLIPVisionConfig
from transformers import AutoConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip.modeling_clip import *
from transformers.utils import replace_return_docstrings, add_start_docstrings_to_model_forward
from model.util.pos_embed import get_2d_sincos_pos_embed


class TactileVideoMAE(nn.Module):
    def __init__(self, args, config, num_frames, tube_size):
        super(TactileVideoMAE, self).__init__()

        config.vision_config.num_frames = num_frames
        config.vision_config.tube_size = tube_size

        config.vision_config.mask_ratio = args.mask_ratio
        config.vision_config.stride = args.stride
        self.stride = stride = args.stride
        self.num_frames = num_frames

        self.num_patches_image = int((config.vision_config.image_size // config.vision_config.patch_size) ** 2 *(num_frames // stride) * (1 - config.vision_config.mask_ratio)) + 1 + 5  # +1 for class token, +5 for sensor token

        print("num_patches_image:", self.num_patches_image)

        self.touch_model = CLIPVisionTransformer(config.vision_config)
        self.touch_projection = nn.Linear(config.vision_config.hidden_size, config.projection_dim, bias=False)
        # self.touch_model.embeddings.embed_dim = 768

        self.num_image_feature_patches = self.touch_model.embeddings.num_patches = int((config.vision_config.image_size // config.vision_config.patch_size) ** 2 * (num_frames // stride))  # num_patches = H * W
        self.touch_model.embeddings.patch_embedding = nn.Conv3d(
            in_channels=config.vision_config.num_channels,
            out_channels=self.touch_model.embeddings.embed_dim,
            kernel_size=(stride, self.touch_model.embeddings.patch_size, self.touch_model.embeddings.patch_size),
            stride=(stride, self.touch_model.embeddings.patch_size, self.touch_model.embeddings.patch_size),
            bias=False,
        )
        self.touch_model.embeddings.position_embedding = nn.Embedding(self.num_image_feature_patches + 1, self.touch_model.embeddings.embed_dim)
        self.patch_size = config.vision_config.patch_size

        self.sensor_token = nn.Parameter(torch.zeros(20, 5, config.vision_config.hidden_size))

        self.touch_model.forward = self.touch_forward
        self.touch_model.embeddings.forward = self.emb_forward

        self.new_position_ids = torch.nn.Parameter(
            torch.arange(self.num_image_feature_patches + 1, dtype=torch.int64).unsqueeze(0), requires_grad=False
        )
        # torch.arange(self.num_image_feature_patches + 1, dtype=torch.int64).unsqueeze(0)  # (1, L)


    @add_start_docstrings_to_model_forward(CLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPVisionConfig)
    def touch_forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        sensor_type = None,
        use_mask = True,
        probe = False
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        # a = self.sensor_token[sensor_type]
        # print(a.shape)
        output_attentions = output_attentions if output_attentions is not None else self.touch_model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.touch_model.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.touch_model.config.use_return_dict

        # if pixel_values is None:
        #     raise ValueError("You have to specify pixel_values")

        hidden_states = self.touch_model.embeddings(pixel_values, sensor_type = sensor_type, use_mask = use_mask)
        hidden_states = self.touch_model.pre_layrnorm(hidden_states)


        attention_mask = None     

        encoder_outputs = self.touch_model.encoder(
            inputs_embeds=hidden_states,
            attention_mask = attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            # return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]

        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.touch_model.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def emb_forward(self, pixel_values: Optional[torch.FloatTensor] = None, noise=None, sensor_type=None, use_mask = True) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        target_dtype = self.touch_model.embeddings.patch_embedding.weight.dtype

        # print(pixel_values.shape) (B, C, T, H, W)
        patch_embeds = self.touch_model.embeddings.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
        # print(patch_embeds.shape) (B, D, T, grid, grid)

        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        pos_emb = self.touch_model.embeddings.position_embedding(self.new_position_ids)  # [1, L, D]

        img_embeddings = patch_embeds + pos_emb[:, 1:, :]
 
        x_masked = img_embeddings

        class_embeds = self.touch_model.embeddings.class_embedding + pos_emb[:, 0, :]
        class_embeds = class_embeds.expand(batch_size, 1, -1)


        sensor_emb = self.sensor_token[sensor_type]
        img_embeddings = torch.cat([class_embeds, sensor_emb, x_masked], dim=1)
        

        return img_embeddings

    def forward(self, x=None, sensor_type=None, probe=False, get_cls=False):
        # print(sensor_type)
        if x is not None and len(x.shape) == 4:
            x = x.unsqueeze(2).repeat(1, 1, self.num_frames, 1, 1)
        elif x is not None and x.shape[1]!= 3:
            x = x.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W) -> (B, C, T, H, W)


        latent = self.forward_encoder(x=x, sensor_type=sensor_type, use_mask=False, get_cls=get_cls, probe=probe)
        return latent


    def forward_encoder(self, x=None, sensor_type=None, use_mask = False, get_cls = False, probe=False):

        if len(x.shape) == 4:
            x = x.unsqueeze(1).repeat(1, self.num_frames, 1, 1, 1)

        x = self.touch_model(x, sensor_type=sensor_type, use_mask = use_mask, probe=probe)
        if get_cls:
            out = self.touch_projection(x.pooler_output)
        elif probe:
            out = x.last_hidden_state
        else:
            out = self.touch_model.post_layernorm(x.last_hidden_state)
            out = self.touch_projection(out)

        return out

