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
from transformers.models.vit.modeling_vit import ViTLayer
from transformers.utils import replace_return_docstrings, add_start_docstrings_to_model_forward
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from sparsh.tactile_ssl.model.util.pos_embed import get_2d_sincos_pos_embed

aaa = {'NUM_FRAMES': 1, 'PATCH_DROPOUT': 0.0}

def set_global_value(k, v):
    global aaa
    aaa[k] = v

def get_global_value():
    global aaa
    return aaa

class PointTokenizer(nn.Module):
    def __init__(self, args, config):
        super(PointTokenizer, self).__init__()
        self.config = config

        self.input_dim = 6  # x, y, z, nx, ny, nz
        self.output_dim = config.hidden_size

        if '-' in args.point_mlp:
            self.hidden_mlp = [int(x) for x in args.point_mlp.split('-')] + [self.output_dim]
        else:
            self.hidden_mlp = [int(args.point_mlp)] + [self.output_dim]
        # self.point_embedding = nn.Linear(6, config.hidden_size, bias=False)
        # self.position_embedding = nn.Embedding(config.num_positions, config.hidden_size)
        # self.class_embedding = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.point_embedding = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_mlp[0]),
            nn.LayerNorm(self.hidden_mlp[0]),
            nn.GELU(),
            *[
                nn.Sequential(
                    nn.Linear(self.hidden_mlp[i], self.hidden_mlp[i + 1]),
                    nn.LayerNorm(self.hidden_mlp[i + 1]),
                    nn.GELU()
                ) for i in range(len(self.hidden_mlp) - 1)
            ]
        )

        self.class_embedding = nn.Parameter(torch.randn(self.output_dim))

    def forward(self, points):
        
        points_emb = self.point_embedding(points)  # [B, N, D]
        class_emb = self.class_embedding.expand(points_emb.shape[0], 1, -1)  # [B, 1, D]
        embeddings = torch.cat([points_emb, class_emb], dim=1)  # [B, N+1, D]

        return embeddings

class CLIPPointEncoderLayer(nn.Module):
    def __init__(self, config: Union[CLIPVisionConfig, CLIPTextConfig]):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = CLIPMLP(config)
        self.mlp_point = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.layer_norm2_point = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.use_sensor_token = config.use_sensor_token
        

        if config.use_sensor_token:
            self.num_patches_image = int((config.image_size // config.patch_size) ** 2 * (config.num_frames // config.stride) * (1 - config.mask_ratio)) + 1 + 5  # +1 for class token, +5 for sensor token
        else:
            self.num_patches_image = int((config.image_size // config.patch_size) ** 2 * (config.num_frames // config.stride) * (1 - config.mask_ratio)) + 1  # +1 for class token

        self.use_point_expert = config.use_point_expert

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
        img_len = None
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states

        # print(img_len, hidden_states.shape)

        if self.use_point_expert:

            if img_len > 0:
                hidden_states_image = hidden_states[:, :img_len, :]
            else:
                hidden_states_image = None
            
            if hidden_states.shape[1] > img_len:
                if img_len > 0:
                    if self.use_sensor_token:
                        hidden_states_point = torch.cat([hidden_states[:, img_len:, :], hidden_states[:, 1:6, :]], dim=1)
                    else:
                        hidden_states_point = hidden_states[:, img_len:, :]
                else:
                    hidden_states_point = hidden_states

            else:
                hidden_states_point = None

            if hidden_states_image is not None:
                hidden_states_image = self.layer_norm2(hidden_states_image)
                hidden_states_image = self.mlp(hidden_states_image)

            if hidden_states_point is not None:
                hidden_states_point = self.layer_norm2_point(hidden_states_point)
                hidden_states_point = self.mlp_point(hidden_states_point)

            if hidden_states_image is None:
                hidden_states = hidden_states_point
            elif hidden_states_point is None:
                hidden_states = hidden_states_image
            else:
                hidden_states = torch.cat([hidden_states_image[:, :1, :], (hidden_states_image[:, 1:6, :] + hidden_states_point[:, -5:, :]) / 2.0, hidden_states_image[:, 6:, :], hidden_states_point[:, :-5, :]], dim=1)

            # print("hidden_states shape:", hidden_states.shape)
            # print("hidden_states_point shape:", hidden_states_point.shape if hidden_states_point is not None else "no point")
            # print("hidden_states_image shape:", hidden_states_image.shape if hidden_states_image is not None else "no image")

        else:
            hidden_states = self.layer_norm2(hidden_states)
            hidden_states = self.mlp(hidden_states)

        # print("hidden_states shape:", hidden_states.shape)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs

class CLIPPointEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`CLIPEncoderLayer`].

    Args:
        config: CLIPConfig
    """

    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([CLIPPointEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        img_len = None
    ) -> BaseModelOutput:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            causal_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Causal mask for the text model. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions,
                    img_len=img_len
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions=output_attentions,
                    img_len=img_len
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )

class TactilePointMAE(nn.Module):
    def __init__(self, args, config, decoder_config, num_frames, add_time_attn, tube_size):
        super(TactilePointMAE, self).__init__()

        config.vision_config.num_frames = num_frames
        config.vision_config.tube_size = tube_size
        config.vision_config.use_point = args.use_point
        config.vision_config.use_point_expert = args.use_point_expert and args.use_point
        config.vision_config.use_sensor_token = args.use_sensor_token
        config.vision_config.max_point_token = args.max_point_token
        config.vision_config.use_image_with_marker = args.use_image_with_marker
        config.vision_config.mask_ratio = args.mask_ratio
        config.vision_config.stride = 1

        if args.use_sensor_token:
            self.num_patches_image = int((config.vision_config.image_size // config.vision_config.patch_size) ** 2 * (1 - config.vision_config.mask_ratio)) + 1 + 5  # +1 for class token, +5 for sensor token
        else:
            self.num_patches_image = int((config.vision_config.image_size // config.vision_config.patch_size) ** 2 * (1 - config.vision_config.mask_ratio)) + 1  # +1 for class token

        print("num_patches_image:", self.num_patches_image)

        self.use_sensor_token = args.use_sensor_token
        self.use_same_patchemb = args.use_same_patchemb
        self.new_decoder_sensor_token = args.new_decoder_sensor_token
        self.use_image_with_marker = args.use_image_with_marker

        self.norm_pix_loss = False
        if args.norm_pix_loss:
            self.norm_pix_loss = args.norm_pix_loss

        self.point_tokenizer = PointTokenizer(args, config.vision_config)

        self.touch_model = CLIPVisionTransformer(config.vision_config)
        self.touch_model.encoder = CLIPPointEncoder(config.vision_config)
        self.touch_projection = nn.Linear(config.vision_config.hidden_size, config.projection_dim, bias=False)

        self.video_patch_embedding = nn.Conv3d(
            in_channels=config.vision_config.num_channels,
            out_channels=self.touch_model.embeddings.embed_dim,
            kernel_size=(3, self.touch_model.embeddings.patch_size, self.touch_model.embeddings.patch_size),
            stride=(3, self.touch_model.embeddings.patch_size, self.touch_model.embeddings.patch_size),
            bias=False,
        )
        self.video_position_embedding = nn.Embedding(self.touch_model.embeddings.num_positions, self.touch_model.embeddings.embed_dim)
        
        self.decoder_embed = nn.Linear(config.projection_dim, decoder_config.hidden_size, bias=True)
        self.num_image_feature_patches = self.touch_model.embeddings.num_patches
        # print("num_image_feature_patches:", self.num_image_feature_patches)
        self.patch_size = config.vision_config.patch_size
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_image_feature_patches + 1, decoder_config.hidden_size), requires_grad=False)
        self.touch_decoder_blocks = nn.ModuleList([CLIPEncoderLayer(decoder_config) for _ in range(decoder_config.num_hidden_layers)])

        self.decoder_norm = nn.LayerNorm(decoder_config.hidden_size, eps=decoder_config.layer_norm_eps)
        self.decoder_pred = nn.Linear(decoder_config.hidden_size, decoder_config.patch_size**2 * decoder_config.num_channels, bias=True)

        self.decoder_pred_video = nn.Linear(decoder_config.hidden_size, decoder_config.patch_size**2 * decoder_config.num_channels * 4, bias=True)

        self.point_decoder_embed = nn.Linear(config.projection_dim, decoder_config.hidden_size, bias=True)
        self.num_points = args.max_point_token
        self.point_decoder_blocks = nn.ModuleList([CLIPEncoderLayer(decoder_config) for _ in range(decoder_config.num_hidden_layers)])
        self.point_decoder_norm = nn.LayerNorm(decoder_config.hidden_size, eps=decoder_config.layer_norm_eps)
        self.point_decoder_pred = nn.Linear(decoder_config.hidden_size, 6, bias=True)

        self.point_decoder_pred_video = nn.Linear(decoder_config.hidden_size, 6 * 4, bias=True)
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_config.hidden_size))
        self.mask_token_point = nn.Parameter(torch.zeros(1, 1, decoder_config.hidden_size))

        if self.use_sensor_token:
            self.sensor_token = nn.Parameter(torch.zeros(20, 5, config.vision_config.hidden_size))
            self.beta = 1.0
            if self.new_decoder_sensor_token:
                self.sensor_token_proj = nn.Linear(config.vision_config.hidden_size, decoder_config.hidden_size, bias=False)

        self.mask_ratio = args.mask_ratio
        self.mask_ratio_point = args.mask_ratio_point

        self.touch_model.forward = self.touch_forward
        self.touch_model.embeddings.forward = self.emb_forward

    
    def initialize_decoder(self):
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.num_image_feature_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # print(decoder_pos_embed.shape, self.decoder_pos_embed.shape, int(self.num_image_feature_patches**.5))
        # exit(0)

        torch.nn.init.normal_(self.mask_token, std=.02)
        if self.use_sensor_token:
            torch.nn.init.normal_(self.sensor_token, std=.02)

    def random_masking(self, sequence, noise=None, points_num=None):
        """
        Perform per-sample random masking by per-sample shuffling. Per-sample shuffling is done by argsort random
        noise.

        Args:
            sequence (`torch.LongTensor` of shape `(batch_size, sequence_length, dim)`)
            noise (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) which is
                mainly used for testing purposes to control randomness and maintain the reproducibility
        """
        batch_size, seq_length, dim = sequence.shape

        if points_num is not None:
            len_keep = int(seq_length * (1 - self.mask_ratio_point))
        else:
            len_keep = int(seq_length * (1 - self.mask_ratio))

        if noise is None:
            noise = torch.rand(batch_size, seq_length, device=sequence.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1).to(sequence.device)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1).to(sequence.device)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        sequence_unmasked = torch.gather(sequence, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))

        if points_num is not None:
            valid_mask = (ids_keep < points_num).int()
        else:
            valid_mask = None

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([batch_size, seq_length], device=sequence.device)
        mask[:, :len_keep] = 0

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        if points_num is not None:
            range_idx = torch.arange(seq_length, device=sequence.device).unsqueeze(0)  # (1, L)
            valid_pos_mask = (range_idx < points_num)  # (B, L)
            mask = mask * valid_pos_mask.int()

            # print("points_num:", points_num)
            # print('mask:', mask)
            # print('ids_keep:', ids_keep)
            # print('valid_mask:', valid_mask)
            # print('ids_restore:', ids_restore)


        # if points_num is not None:
        #     print(sequence_unmasked.shape)
        #     print(valid_mask)


        return sequence_unmasked, mask, ids_restore, valid_mask

    @add_start_docstrings_to_model_forward(CLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPVisionConfig)
    def touch_forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        points: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        sensor_type = None,
        data_type = None,
        use_mask = True,
        points_num = None
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

        hidden_states, mask_img, mask_pts, ids_restore_img, ids_restore_pts, valid_mask, img_len  = self.touch_model.embeddings(pixel_values, points=points, points_num=points_num, sensor_type = sensor_type, data_type = data_type, use_mask = use_mask)
        hidden_states = self.touch_model.pre_layrnorm(hidden_states)

        # print("hidden_states shape:", hidden_states.shape)

        attention_mask = None
        if points_num is not None:
            attention_mask = torch.zeros((hidden_states.shape[0], hidden_states.shape[1]), device=hidden_states.device)
            
            attention_mask[:, :img_len] = 1  # +6 for class token and sensor tokens
            attention_mask[:, -1] = 1

            # print(hidden_states.shape[1], img_len, img_len+valid_mask.shape[1])
            attention_mask[:, img_len:img_len+valid_mask.shape[1]] = valid_mask[:, :]  # valid mask for points
            # print(img_len)
            # print(valid_mask)
            # print(attention_mask)

            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)
            # print(attention_mask)       

        encoder_outputs = self.touch_model.encoder(
            inputs_embeds=hidden_states,
            attention_mask = attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            img_len = img_len
            # return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        if points_num is not None:
            pooled_output = (last_hidden_state[:, 0, :] + last_hidden_state[:, -1, :]) / 2.0
            pooled_output = self.touch_model.post_layernorm(pooled_output)
        else:
            pooled_output = last_hidden_state[:, 0, :]
            pooled_output = self.touch_model.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        ), mask_img, mask_pts, ids_restore_img, ids_restore_pts

    def emb_forward(self, pixel_values: Optional[torch.FloatTensor] = None, points=None, points_num=None, noise=None, sensor_type=None, data_type = None, use_mask = True) -> torch.Tensor:
        if pixel_values is not None:
            batch_size = pixel_values.shape[0]
            target_dtype = self.touch_model.embeddings.patch_embedding.weight.dtype
            if data_type == 0 and (not self.use_same_patchemb):
                patch_embeds = self.touch_model.embeddings.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
            else:
                patch_embeds = self.video_patch_embedding(pixel_values.to(dtype=target_dtype))

            patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

            pos_emb = self.touch_model.embeddings.position_embedding(self.touch_model.embeddings.position_ids)

            img_embeddings = patch_embeds + pos_emb[:, 1:, :]
            if use_mask:
                x_masked, mask_img, ids_restore_img, valid_lens_img = self.random_masking(img_embeddings, noise)
            else:
                x_masked = img_embeddings
                mask_img = torch.ones(1)
                ids_restore_img = torch.ones(1)
                valid_lens_img = None

            class_embeds = self.touch_model.embeddings.class_embedding + pos_emb[:, 0, :]
            class_embeds = class_embeds.expand(batch_size, 1, -1)

            if self.use_sensor_token:
                sensor_emb = self.sensor_token[sensor_type]
                img_embeddings = torch.cat([class_embeds, sensor_emb, x_masked], dim=1)
            else:
                img_embeddings = torch.cat([class_embeds, x_masked], dim=1)
        
        if points is not None:
            point_embeddings = self.point_tokenizer(points)  # [B, N, D]

            if use_mask:
                pts_embeddings, mask_pts, ids_restore_pts, valid_lens_pts = self.random_masking(point_embeddings[:,:-1], noise, points_num)
                if pixel_values is not None:
                    pts_embeddings = torch.cat([pts_embeddings, point_embeddings[:,-1:]], dim=1)
                else:
                    sensor_emb = self.sensor_token[sensor_type]
                    pts_embeddings = torch.cat([sensor_emb, pts_embeddings, point_embeddings[:,-1:]], dim=1)  # keep the last point
            else:
                mask_pts = torch.ones(1)
                ids_restore_pts = torch.ones(1)
            
        if points is not None and pixel_values is not None:
            embeddings = torch.cat([img_embeddings, pts_embeddings], dim=1)
            # mask = torch.cat([mask_img, mask_pts], dim=1)
            # ids_restore = torch.cat([ids_restore_img, ids_restore_pts + x_masked.shape[1]], dim=1)  # offset pts index
            valid_lens = valid_lens_pts
            img_len = img_embeddings.shape[1]
        
        elif points is not None:
            embeddings = pts_embeddings
            mask_img = None
            ids_restore_img = None
            valid_lens = valid_lens_pts
            img_len = 0
        
        elif pixel_values is not None:
            embeddings = img_embeddings
            mask_pts = None
            ids_restore_pts = None
            valid_lens = valid_lens_img
            img_len = img_embeddings.shape[1]
        
        else:
            raise ValueError("Either pixel_values or points must be provided")


        return embeddings, mask_img, mask_pts, ids_restore_img, ids_restore_pts, valid_lens, img_len

    def forward(self, x=None, points=None, points_num=None, sensor_type=None, data_type=None, target_sensor_type = None):
        if data_type == 0:
            latent, mask_img, mask_pts, ids_restore_img, ids_restore_pts = self.forward_encoder(x=x, points=points, points_num=points_num, sensor_type=sensor_type, data_type=data_type)
        else:
            latent, mask_img, mask_pts, ids_restore_img, ids_restore_pts = self.forward_encoder(x=x[:, :3], points=points, points_num=points_num, sensor_type=sensor_type, data_type=data_type)
        
        # print(mask_img.shape if mask_img is not None else "no mask_img")
        # if self.use_sensor_token:
        #     add_token_num = 6  # 1 for cls token, 5 for sensor tokens
        # else:
        #     add_token_num = 1
        if points is not None:
            if self.use_image_with_marker and mask_img is not None:
                if self.use_sensor_token:
                    points_latent = torch.cat([latent[:, -1:, :], latent[:, 1:6, :], latent[:, self.num_patches_image:-1, :]], dim=1)
                else:
                    points_latent = torch.cat([latent[:, -1:, :], latent[:, self.num_patches_image:-1, :]], dim=1)
                img_latent = latent[:, :self.num_patches_image, :]

            else:

                points_latent = torch.cat([latent[:, -1:, :], latent[:, :-1, :]], dim=1)
                img_latent = None

        else:
            points_latent = None
            img_latent = latent

        # print(img_latent.shape if img_latent is not None else "no image latent")
        # print(points_latent.shape if points_latent is not None else "no points latent")
        pred_img = torch.zeros(1, device=latent.device)
        loss_img = torch.zeros(1, device=latent.device)
        if img_latent is not None:
            pred_img = self.forward_decoder(img_latent, ids_restore_img, data_type=data_type, sensor_type=target_sensor_type)
            loss_img = self.forward_loss_img(x, pred_img, mask_img, data_type=data_type)

        pred_pts = torch.zeros(1, device=latent.device)
        loss_pts = torch.zeros(1, device=latent.device)
        if points_latent is not None:
            pred_pts = self.forward_point_decoder(points_latent, points_num, ids_restore_pts, sensor_type=target_sensor_type, data_type=data_type)
            loss_pts = self.forward_loss_pts(points, pred_pts, mask_pts, data_type=data_type)
        
        return loss_img, loss_pts, pred_img, pred_pts, mask_img, mask_pts

    def forward_encoder(self, x=None, points=None, points_num=None, sensor_type=None, data_type = None, use_mask = True):
        if data_type == 0 and self.use_same_patchemb and x is not None:
            x = x.unsqueeze(1).repeat(1, 3, 1, 1, 1)
            # print(x.shape)
        x, mask_img, mask_pts, ids_restore_img, ids_restore_pts = self.touch_model(x, points=points, points_num=points_num, sensor_type=sensor_type, data_type=data_type, use_mask = use_mask)
        if use_mask:
            out = self.touch_projection(x.last_hidden_state)
        else:
            out = self.touch_projection(x.pooler_output)

        return out, mask_img, mask_pts, ids_restore_img, ids_restore_pts
    
    def forward_decoder(self, x, ids_restore, sensor_type=None, data_type = None):

        x = self.decoder_embed(x)

        if self.use_sensor_token:
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 6 - x.shape[1], 1)

            x_ = torch.cat([x[:, 6:, :], mask_tokens], dim=1)  # no cls token and sensor token
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

            if self.new_decoder_sensor_token:
                decoder_sensor = self.sensor_token_proj(self.sensor_token[sensor_type])
                x = torch.cat([x[:, :1, :], decoder_sensor, x_], dim=1)  # append cls token and sensor token
            else:
                x = torch.cat([x[:, :6, :], x_], dim=1)  # append cls token and sensor token
            #x = torch.cat([x[:, :6, :], x_], dim=1)  # append cls token and sensor token

            x[:,0,:] += self.decoder_pos_embed[:,0,:]
            x[:,6:,:] += self.decoder_pos_embed[:,1:,:]
        else:
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)

            x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
            x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

            x = x + self.decoder_pos_embed

        for blk in self.touch_decoder_blocks:
            layer_outputs = blk(x, attention_mask=None, causal_attention_mask=None, output_attentions=False)
        
            x = layer_outputs[0]

        x = self.decoder_norm(x)

        if data_type == 0:
            x = self.decoder_pred(x)
        else:
            x = self.decoder_pred_video(x)

        if self.use_sensor_token:
            x = x[:, 6:, :]
        else:
            x = x[:, 1:, :]

        if data_type == 1:
            x = x.view(x.shape[0], x.shape[1], 4, -1)
        return x
    
    def forward_point_decoder(self, points, points_num, ids_restore, sensor_type=None, data_type = None):

        points = self.point_decoder_embed(points)

        if self.use_sensor_token:
            mask_tokens = self.mask_token_point.repeat(points.shape[0], ids_restore.shape[1] + 6 - points.shape[1], 1)
            points_ = torch.cat([points[:, 6:, :], mask_tokens], dim=1)  # no cls token and sensor token
            points_ = torch.gather(points_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, points.shape[2]))  # unshuffle
            points = torch.cat([points[:, :6, :], points_], dim=1)  # append cls token and sensor token

        else:
            mask_tokens = self.mask_token_point.repeat(points.shape[0], ids_restore.shape[1] + 1 - points.shape[1], 1)
            points_ = torch.cat([points[:, 1:, :], mask_tokens], dim=1)  # no cls token
            points_ = torch.gather(points_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, points.shape[2]))  # unshuffle
            points = torch.cat([points[:, :1, :], points_], dim=1)  # append cls token

        attention_mask = torch.zeros((points.shape[0], points.shape[1]), device=points.device)

        range_idx = torch.arange(points.shape[1], device=points.device).unsqueeze(0)  # (1, L)
        if self.use_sensor_token:
            valid_pos_mask = (range_idx < points_num + 6)  # (B, L)
        else:
            valid_pos_mask = (range_idx < points_num + 1)  # (B, L)

        attention_mask = attention_mask + valid_pos_mask.int()

        # print(attention_mask)

        attention_mask = _prepare_4d_attention_mask(attention_mask, points.dtype)
        

        for blk in self.point_decoder_blocks:
            layer_outputs = blk(points, attention_mask=attention_mask, causal_attention_mask=None, output_attentions=False)
        
            points = layer_outputs[0]
        
        points = self.point_decoder_norm(points)

        points = self.point_decoder_pred(points)

        if self.use_sensor_token:
            points = points[:, 6:, :]
        else:
            points = points[:, 1:, :]  # remove cls token

        return points
        

    
    def forward_loss_img(self, x, pred, mask, data_type = None):
        target = self.patchify(x, data_type = data_type)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        
        if data_type == 0:
            loss = (pred - target) ** 2
            loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

            loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        else:
            loss = (pred - target) ** 2
            loss_pred = loss[:, :, 3, :].mean(dim=-1)
            loss_recon = loss[:, :, :3, :].mean(dim=-2).mean(dim=-1)
            L = loss.shape[1] * loss.shape[0]
            loss = (loss_recon * mask).sum() / mask.sum() + loss_pred.sum() / L
        
        return loss
    
    def forward_loss_pts(self, points, pred, mask, data_type = None):
        # print(pred, points)
        # print(mask)
        loss = (pred - points) ** 2
        # print(loss)
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        # print(loss)
        # print(mask.shape,mask.sum())
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        # print(loss)
        return loss

    def patchify(self, imgs, data_type = None):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        if data_type == 0:
            # image
            p = self.patch_size
            assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

            h = w = imgs.shape[2] // p
            x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
            x = torch.einsum('nchpwq->nhwpqc', x)
            x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        else:
            # video
            p = self.patch_size
            assert imgs.shape[3] == imgs.shape[4] and imgs.shape[3] % p == 0

            h = w = imgs.shape[3] // p
            x = imgs.reshape(shape=(imgs.shape[0], 4, 3, h, p, w, p))
            x = torch.einsum('ntchpwq->nhwtpqc', x)
            x = x.reshape(shape=(imgs.shape[0], h * w, 4, p**2 * 3))
        return x

    def unpatchify(self, x, data_type = None):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        if data_type == 0:
            # image
            p = self.patch_size
            h = w = int(x.shape[1]**.5)
            assert h * w == x.shape[1]
            
            x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
            x = torch.einsum('nhwpqc->nchpwq', x)
            imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))

        else:
            p = self.patch_size
            h = w = int(x.shape[1]**.5)
            assert h * w == x.shape[1]

            x = x.reshape(shape=(x.shape[0], h, w, 4, p, p, 3))
            x = torch.einsum('nhwtpqc->ntchpwq', x)
            imgs = x.reshape(shape=(x.shape[0], 4, 3, h * p, h * p))
        return imgs

class TactileVideoPointMAE(nn.Module):
    def __init__(self, config, decoder_config, num_frames, add_time_attn, tube_size, now_sensor='gelsight'):
        super(TactileVideoPointMAE, self).__init__()

        config.vision_config.num_frames = num_frames
        config.vision_config.tube_size = tube_size
        config.vision_config.use_point = False
        config.vision_config.use_point_expert = False
        config.vision_config.use_sensor_token = True
        config.vision_config.max_point_token = 200
        config.vision_config.use_image_with_marker = True
        config.vision_config.mask_ratio = 0.0
        config.vision_config.stride = 2
        self.stride = stride = 2
        self.num_frames = num_frames
        self.use_diff = False

        # config.vision_config.hidden_size = 768
        # config.vision_config.intermediate_size = 3072
        # print((config.vision_config.image_size // config.vision_config.patch_size) ** 2 *(num_frames // stride))

        self.num_patches_image = int((config.vision_config.image_size // config.vision_config.patch_size) ** 2 *(num_frames // stride) * (1 - config.vision_config.mask_ratio)) + 1 + 5  # +1 for class token, +5 for sensor token

        print("num_patches_image:", self.num_patches_image)

        self.use_sensor_token = True
        # self.use_same_patchemb = args.use_same_patchemb
        # self.new_decoder_sensor_token = args.new_decoder_sensor_token
        self.use_image_with_marker = True

        self.norm_pix_loss = False

        # self.point_tokenizer = PointTokenizer(args, config.vision_config)

        self.touch_model = CLIPVisionTransformer(config.vision_config)
        self.touch_model.encoder = CLIPPointEncoder(config.vision_config)
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
        # self.video_position_embedding = nn.Embedding(self.touch_model.embeddings.num_positions, self.touch_model.embeddings.embed_dim)
        
        if decoder_config is not None:
            self.decoder_embed = nn.Linear(config.projection_dim, decoder_config.hidden_size, bias=True)
            self.decoder_embed_diff = nn.Linear(config.projection_dim, decoder_config.hidden_size, bias=True)
            # self.num_image_feature_patches = self.touch_model.embeddings.num_patches = int((config.vision_config.image_size // config.vision_config.patch_size) ** 2 * (num_frames // stride))  # num_patches = H * W
            # print("num_image_feature_patches:", self.num_image_feature_patches)
            self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_image_feature_patches + 1, decoder_config.hidden_size), requires_grad=False)

            self.touch_decoder_blocks = nn.ModuleList([CLIPEncoderLayer(decoder_config) for _ in range(decoder_config.num_hidden_layers)])
            self.decoder_norm = nn.LayerNorm(decoder_config.hidden_size, eps=decoder_config.layer_norm_eps)
            self.decoder_pred_video = nn.Linear(decoder_config.hidden_size, decoder_config.patch_size**2 * decoder_config.num_channels * stride, bias=True)

            self.diff_touch_decoder_blocks = nn.ModuleList([CLIPEncoderLayer(decoder_config) for _ in range(decoder_config.num_hidden_layers)])
            self.diff_decoder_norm = nn.LayerNorm(decoder_config.hidden_size, eps=decoder_config.layer_norm_eps)
            self.diff_decoder_pred_video = nn.Linear(decoder_config.hidden_size, decoder_config.patch_size**2 * decoder_config.num_channels * stride, bias=True)

            self.point_decoder_embed = nn.Linear(config.projection_dim, decoder_config.hidden_size, bias=True)
            self.num_points = 200
            self.point_decoder_blocks = nn.ModuleList([CLIPEncoderLayer(decoder_config) for _ in range(decoder_config.num_hidden_layers)])
            self.point_decoder_norm = nn.LayerNorm(decoder_config.hidden_size, eps=decoder_config.layer_norm_eps)
            self.point_decoder_pred = nn.Linear(decoder_config.hidden_size, 6, bias=True)

            # self.point_decoder_pred_video = nn.Linear(decoder_config.hidden_size, 6 * stride, bias=True)
            
            self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_config.hidden_size))
            self.mask_token_diff = nn.Parameter(torch.zeros(1, 1, decoder_config.hidden_size))
            self.mask_token_point = nn.Parameter(torch.zeros(1, 1, decoder_config.hidden_size))

        if self.use_sensor_token:
            self.sensor_token = nn.Parameter(torch.zeros(20, 5, config.vision_config.hidden_size))
            self.beta = 1.0
            # if self.new_decoder_sensor_token:
            #     self.sensor_token_proj = nn.Linear(config.vision_config.hidden_size, decoder_config.hidden_size, bias=False)

        self.mask_ratio = 0.75
        self.mask_ratio_point = 0.5

        self.touch_model.forward = self.touch_forward
        self.touch_model.embeddings.forward = self.emb_forward

        self.new_position_ids = torch.nn.Parameter(
            torch.arange(self.num_image_feature_patches + 1, dtype=torch.int64).unsqueeze(0), requires_grad=False
        )

        if now_sensor == 'gelsight':
            self.now_sensor = torch.tensor([3]*1000, dtype=torch.int64)
        elif now_sensor=='digit':
            self.now_sensor = torch.tensor([1]*1000, dtype=torch.int64)

        # torch.arange(self.num_image_feature_patches + 1, dtype=torch.int64).unsqueeze(0)  # (1, L)

    def del_decoder(self): 
        self.decoder_embed = None
        self.decoder_pos_embed = None
        self.touch_decoder_blocks = None
        self.decoder_norm = None
        self.decoder_pred_video = None

        self.diff_decoder_embed = None
        self.diff_decoder_pos_embed = None
        self.diff_touch_decoder_blocks = None
        self.diff_decoder_norm = None
        self.diff_decoder_pred_video = None

        self.point_decoder_embed = None
        self.point_decoder_blocks = None
        self.point_decoder_norm = None
        self.point_decoder_pred = None
    
    def initialize_decoder(self):
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int((self.num_image_feature_patches // ((self.num_frames // self.stride)))**.5), cls_token=True)
        decoder_pos_embed = torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        new_decoder_pos_embed = [decoder_pos_embed[:, :1, :]] + [decoder_pos_embed[:, 1:, :] for _ in range(self.num_frames // self.stride)]
        new_decoder_pos_embed = torch.cat(new_decoder_pos_embed, dim=1)
        self.decoder_pos_embed.data.copy_(new_decoder_pos_embed)

        torch.nn.init.normal_(self.mask_token, std=.02)
        if self.use_sensor_token:
            torch.nn.init.normal_(self.sensor_token, std=.02)

    def random_masking(self, sequence, noise=None, points_num=None):

        batch_size, seq_length, dim = sequence.shape
        sub_sequence = seq_length // (self.num_frames // self.stride)

        if points_num is not None:
            len_keep = int(sub_sequence * (1 - self.mask_ratio_point))
        else:
            len_keep = int(sub_sequence * (1 - self.mask_ratio))

        if noise is None:
            noise = torch.rand(batch_size, sub_sequence, device=sequence.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1).to(sequence.device)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1).to(sequence.device)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]

        ids_keep_full = torch.cat([ ids_keep.clone() + sub_sequence * i for i in range(self.num_frames // self.stride)], dim=1)  # repeat ids_keep for each frame
        ids_restore_full = torch.cat([ ids_restore.clone() + sub_sequence * i for i in range(self.num_frames // self.stride)], dim=1)  # repeat ids_restore for each frame
        # print(ids_keep.shape, ids_restore.shape, sequence.shape)
        sequence_unmasked = torch.gather(sequence, dim=1, index=ids_keep_full.unsqueeze(-1).repeat(1, 1, dim))

        if points_num is not None:
            valid_mask = (ids_keep < points_num).int()
            valid_mask = valid_mask.repeat(1, self.num_frames // self.stride)  # repeat valid mask for each frame
        else:
            valid_mask = None

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([batch_size, seq_length], device=sequence.device)
        mask[:, :len_keep] = 0
        # mask = mask.repeat(1, self.num_frames // self.stride)  # repeat mask for each frame

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        if points_num is not None:
            range_idx = torch.arange(sub_sequence, device=sequence.device).unsqueeze(0)  # (1, L)
            valid_pos_mask = (range_idx < points_num)  # (B, L)
            valid_pos_mask = valid_pos_mask.repeat(1, self.num_frames // self.stride)  # repeat valid mask for each frame
            mask = mask * valid_pos_mask.int()

        return sequence_unmasked, mask, ids_restore_full, valid_mask

    @add_start_docstrings_to_model_forward(CLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPVisionConfig)
    def touch_forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        points: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        sensor_type = None,
        data_type = 1,
        use_mask = True,
        points_num = None,
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

        hidden_states, mask_img, mask_pts, ids_restore_img, ids_restore_pts, valid_mask, img_len  = self.touch_model.embeddings(pixel_values, points=points, points_num=points_num, sensor_type = sensor_type, data_type = data_type, use_mask = use_mask)
        hidden_states = self.touch_model.pre_layrnorm(hidden_states)

        # print("hidden_states shape:", hidden_states.shape)

        attention_mask = None   

        encoder_outputs = self.touch_model.encoder(
            inputs_embeds=hidden_states,
            attention_mask = attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            img_len = img_len
            # return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        # last_hidden_state = self.touch_model.post_layernorm(last_hidden_state)

        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.touch_model.post_layernorm(pooled_output)
        

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        ), mask_img, mask_pts, ids_restore_img, ids_restore_pts

    def emb_forward(self, pixel_values: Optional[torch.FloatTensor] = None, points=None, points_num=None, noise=None, sensor_type=None, data_type = None, use_mask = True) -> torch.Tensor:
        if pixel_values is not None:
            batch_size = pixel_values.shape[0]
            target_dtype = self.touch_model.embeddings.patch_embedding.weight.dtype

            # print(pixel_values.shape) (B, C, T, H, W)
            # print(pixel_values.shape)
            patch_embeds = self.touch_model.embeddings.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
            # print(patch_embeds.shape) (B, D, T, grid, grid)

            patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
            # 还原参考********
            # print(patch_embeds.shape) (B, T*grid*grid, D)
            # print(self.touch_model.embeddings.position_ids.shape) (1, 257)
            # print(self.touch_model.embeddings.position_ids.dtype)
            # print(self.new_position_ids.device, self.new_position_ids.dtype)
            # self.new_position_ids.to(patch_embeds.device)

            pos_emb = self.touch_model.embeddings.position_embedding(self.new_position_ids)  # [1, L, D]

            img_embeddings = patch_embeds + pos_emb[:, 1:, :]
            if use_mask:
                x_masked, mask_img, ids_restore_img, valid_lens_img = self.random_masking(img_embeddings, noise)
            else:
                x_masked = img_embeddings
                mask_img = torch.ones(1)
                ids_restore_img = torch.ones(1)
                valid_lens_img = None

            class_embeds = self.touch_model.embeddings.class_embedding + pos_emb[:, 0, :]
            class_embeds = class_embeds.expand(batch_size, 1, -1)

            if self.use_sensor_token:
                sensor_emb = self.sensor_token[sensor_type]
                img_embeddings = torch.cat([class_embeds, sensor_emb, x_masked], dim=1)
            else:
                img_embeddings = torch.cat([class_embeds, x_masked], dim=1)
        
        if points is not None:
            point_embeddings = self.point_tokenizer(points)  # [B, N, D]

            if use_mask:
                pts_embeddings, mask_pts, ids_restore_pts, valid_lens_pts = self.random_masking(point_embeddings[:,:-1], noise, points_num)
                if pixel_values is not None:
                    pts_embeddings = torch.cat([pts_embeddings, point_embeddings[:,-1:]], dim=1)
                else:
                    sensor_emb = self.sensor_token[sensor_type]
                    pts_embeddings = torch.cat([sensor_emb, pts_embeddings, point_embeddings[:,-1:]], dim=1)  # keep the last point
            else:
                mask_pts = torch.ones(1)
                ids_restore_pts = torch.ones(1)
            
        if points is not None and pixel_values is not None:
            embeddings = torch.cat([img_embeddings, pts_embeddings], dim=1)
            # mask = torch.cat([mask_img, mask_pts], dim=1)
            # ids_restore = torch.cat([ids_restore_img, ids_restore_pts + x_masked.shape[1]], dim=1)  # offset pts index
            valid_lens = valid_lens_pts
            img_len = img_embeddings.shape[1]
        
        elif points is not None:
            embeddings = pts_embeddings
            mask_img = None
            ids_restore_img = None
            valid_lens = valid_lens_pts
            img_len = 0
        
        elif pixel_values is not None:
            embeddings = img_embeddings
            mask_pts = None
            ids_restore_pts = None
            valid_lens = valid_lens_img
            img_len = img_embeddings.shape[1]
        
        else:
            raise ValueError("Either pixel_values or points must be provided")


        return embeddings, mask_img, mask_pts, ids_restore_img, ids_restore_pts, valid_lens, img_len

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        sensor_type = self.now_sensor[:x.shape[0]].to(x.device)


        latent, mask_img, mask_pts, ids_restore_img, ids_restore_pts = self.forward_encoder(x=x, sensor_type=sensor_type, use_mask=False, get_cls=False, probe=True)
        # remove the token of first two frames
        # print(latent.shape)
        # latent = torch.cat([latent[:, :6, :], latent[:, self.num_image_feature_patches//2+6:, :]], dim=1)
        # print(latent.shape)
        
        return latent


    def forward_encoder(self, x=None, points=None, points_num=None, sensor_type=None, data_type = None, use_mask = True, get_cls = False, probe=True):

        if len(x.shape) == 4:
            x = x.unsqueeze(1).repeat(1, self.num_frames, 1, 1, 1)

        x, mask_img, mask_pts, ids_restore_img, ids_restore_pts = self.touch_model(x, points=points, points_num=points_num, sensor_type=sensor_type, data_type=data_type, use_mask = use_mask, probe=probe)
        if get_cls:
            out = self.touch_projection(x.pooler_output)
        elif probe:
            out = x.last_hidden_state
            # out = self.touch_projection(x.last_hidden_state)
        else:
            out = self.touch_projection(x.last_hidden_state)

        return out, mask_img, mask_pts, ids_restore_img, ids_restore_pts
    
    def forward_decoder(self, x, ids_restore, sensor_type=None, data_type = None):

        x = self.decoder_embed(x)

        if self.use_sensor_token:
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 6 - x.shape[1], 1)

            x_no_cls = x[:, 6:, :]  # no cls token and sensor token
            length_feature = x_no_cls.shape[1] // (self.num_frames // self.stride)  # length of feature tokens
            length_mask = mask_tokens.shape[1] // (self.num_frames // self.stride) # length of mask tokens
            x_ = []
            # print(x.shape, x_no_cls.shape, mask_tokens.shape, length_feature, length_mask, ids_restore.shape)
            for i in range(self.num_frames // self.stride):
                # print(i*length_feature, (i+1)*length_feature, i*length_mask, (i+1)*length_mask)
                x_.append(x_no_cls[:, i*length_feature:(i+1)*length_feature, :])
                x_.append(mask_tokens[:, i*length_mask:(i+1)*length_mask, :])
            x_ = torch.cat(x_, dim=1)  # no cls token and sensor token
            # print(x_.shape)
            # x_ = torch.cat([x[:, 6:, :], mask_tokens], dim=1)  # no cls token and sensor token
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

            # if self.new_decoder_sensor_token:
            #     decoder_sensor = self.sensor_token_proj(self.sensor_token[sensor_type])
            #     x = torch.cat([x[:, :1, :], decoder_sensor, x_], dim=1)  # append cls token and sensor token
            # else:
            x = torch.cat([x[:, :6, :], x_], dim=1)  # append cls token and sensor token

            #x = torch.cat([x[:, :6, :], x_], dim=1)  # append cls token and sensor token

            x[:,0,:] += self.decoder_pos_embed[:,0,:]
            x[:,6:,:] += self.decoder_pos_embed[:,1:,:]
        else:
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)

            x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
            x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

            x = x + self.decoder_pos_embed

        for blk in self.touch_decoder_blocks:
            layer_outputs = blk(x, attention_mask=None, causal_attention_mask=None, output_attentions=False)
        
            x = layer_outputs[0]

        x = self.decoder_norm(x)


        x = self.decoder_pred_video(x)

        if self.use_sensor_token:
            x = x[:, 6:, :]
        else:
            x = x[:, 1:, :]

        # if data_type == 1:
        #     x = x.view(x.shape[0], x.shape[1], 4, -1)
        # print(x.shape) (B, T//stride*N, patch_size**2 * 3 * stride)
        x = x.view(x.shape[0], self.num_frames // self.stride, -1, x.shape[-1])  
        # print(x.shape) (B, T//stride, N, patch_size**2 * 3 * stride)
        x = x.view(x.shape[0], x.shape[1], x.shape[2], self.stride, -1)
        # x = x.permute(0, 3, 1, 2, 4)
        # x = x.reshape(x.shape[0], x.shape[1]*x.shape[2], x.shape[3], x.shape[4]) 
        x = x.permute(0, 2, 3, 1, 4)  # (B, N, stride, T//stride, patch_size**2 * 3)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2]*x.shape[3], -1)  # (B, N, T, patch_size**2 * 3)
        # print(x.shape) 

        idx = torch.arange(self.num_frames).reshape(self.stride, -1)
        idx = idx.transpose(0, 1).reshape(-1)
        result = x[:, :, idx, :]  # reorder the frames according to the original order
        # print(x.shape)
        # print(result.shape)
        return result

    def forward_diff_decoder(self, x, ids_restore, sensor_type=None, data_type = None):

        x = self.decoder_embed_diff(x)

        if self.use_sensor_token:
            mask_tokens = self.mask_token_diff.repeat(x.shape[0], ids_restore.shape[1] + 6 - x.shape[1], 1)

            x_no_cls = x[:, 6:, :]  # no cls token and sensor token
            length_feature = x_no_cls.shape[1] // (self.num_frames // self.stride)  # length of feature tokens
            length_mask = mask_tokens.shape[1] // (self.num_frames // self.stride) # length of mask tokens
            x_ = []
            for i in range(self.num_frames // self.stride):
                x_.append(x_no_cls[:, i*length_feature:(i+1)*length_feature, :])
                x_.append(mask_tokens[:, i*length_mask:(i+1)*length_mask, :])
            x_ = torch.cat(x_, dim=1)  # no cls token and sensor token
            # x_ = torch.cat([x[:, 6:, :], mask_tokens], dim=1)  # no cls token and sensor token
            # x_ = torch.cat([x_no_cls[:, :] ])
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

            # if self.new_decoder_sensor_token:
            #     decoder_sensor = self.sensor_token_proj(self.sensor_token[sensor_type])
            #     x = torch.cat([x[:, :1, :], decoder_sensor, x_], dim=1)  # append cls token and sensor token
            # else:
            x = torch.cat([x[:, :6, :], x_], dim=1)  # append cls token and sensor token
            #x = torch.cat([x[:, :6, :], x_], dim=1)  # append cls token and sensor token

            x[:,0,:] += self.decoder_pos_embed[:,0,:]
            x[:,6:,:] += self.decoder_pos_embed[:,1:,:]
        else:
            mask_tokens = self.mask_token_diff.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)

            x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
            x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

            x = x + self.decoder_pos_embed

        for blk in self.diff_touch_decoder_blocks:
            layer_outputs = blk(x, attention_mask=None, causal_attention_mask=None, output_attentions=False)
        
            x = layer_outputs[0]

        x = self.diff_decoder_norm(x)

        x = self.diff_decoder_pred_video(x)

        if self.use_sensor_token:
            x = x[:, 6:, :]
        else:
            x = x[:, 1:, :]

        # print(x.shape) (B, T//stride*N, patch_size**2 * 3 * stride)
        x = x.view(x.shape[0], self.num_frames // self.stride, -1, x.shape[-1])  
        # print(x.shape) (B, T//stride, N, patch_size**2 * 3 * stride)
        x = x.view(x.shape[0], x.shape[1], x.shape[2], self.stride, -1)
        # x = x.permute(0, 3, 1, 2, 4)
        # x = x.reshape(x.shape[0], x.shape[1]*x.shape[2], x.shape[3], x.shape[4]) 
        x = x.permute(0, 2, 3, 1, 4)  # (B, N, stride, T//stride, patch_size**2 * 3)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2]*x.shape[3], -1)  # (B, N, T, patch_size**2 * 3)
        # print(x.shape) 

        idx = torch.arange(self.num_frames).reshape(self.stride, -1)
        idx = idx.transpose(0, 1).reshape(-1)
        result = x[:, :, idx, :]  # reorder the frames according to the original order
        result = result[:, :, 1:, :]  # remove the first frame, which is the same as the last frame in the original video
        # print(x.shape)
        # print(result.shape)
        return result
    
    def forward_point_decoder(self, points, points_num, ids_restore, sensor_type=None, data_type = None):

        points = self.point_decoder_embed(points)

        if self.use_sensor_token:
            mask_tokens = self.mask_token_point.repeat(points.shape[0], ids_restore.shape[1] + 6 - points.shape[1], 1)
            points_ = torch.cat([points[:, 6:, :], mask_tokens], dim=1)  # no cls token and sensor token
            points_ = torch.gather(points_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, points.shape[2]))  # unshuffle
            points = torch.cat([points[:, :6, :], points_], dim=1)  # append cls token and sensor token

        else:
            mask_tokens = self.mask_token_point.repeat(points.shape[0], ids_restore.shape[1] + 1 - points.shape[1], 1)
            points_ = torch.cat([points[:, 1:, :], mask_tokens], dim=1)  # no cls token
            points_ = torch.gather(points_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, points.shape[2]))  # unshuffle
            points = torch.cat([points[:, :1, :], points_], dim=1)  # append cls token

        attention_mask = torch.zeros((points.shape[0], points.shape[1]), device=points.device)

        range_idx = torch.arange(points.shape[1], device=points.device).unsqueeze(0)  # (1, L)
        if self.use_sensor_token:
            valid_pos_mask = (range_idx < points_num + 6)  # (B, L)
        else:
            valid_pos_mask = (range_idx < points_num + 1)  # (B, L)

        attention_mask = attention_mask + valid_pos_mask.int()

        # print(attention_mask)

        attention_mask = _prepare_4d_attention_mask(attention_mask, points.dtype)
        

        for blk in self.point_decoder_blocks:
            layer_outputs = blk(points, attention_mask=attention_mask, causal_attention_mask=None, output_attentions=False)
        
            points = layer_outputs[0]
        
        points = self.point_decoder_norm(points)

        points = self.point_decoder_pred(points)

        if self.use_sensor_token:
            points = points[:, 6:, :]
        else:
            points = points[:, 1:, :]  # remove cls token

        return points
        

    
    def forward_loss_img(self, x, pred, mask, data_type = None):
        target = self.patchify(x, data_type = data_type)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        
        loss = (pred - target) ** 2

        # print(pred.shape, target.shape, loss.shape, mask.shape, 'sss')

        # loss_pred = loss[:, :, 3, :].mean(dim=-1)
        # loss_recon = loss[:, :, :3, :].mean(dim=-2).mean(dim=-1)
        # L = loss.shape[1] * loss.shape[0]
        # loss = (loss_recon * mask).sum() / mask.sum() + loss_pred.sum() / L
        loss = loss.mean(dim=-2).mean(dim=-1)
        L = loss.shape[1] * loss.shape[0]
        loss = (loss * mask).sum() / mask.sum()
        
        return loss
    
    def forward_loss_pts(self, points, pred, mask, data_type = None):
        # print(pred, points)
        # print(mask)
        loss = (pred - points) ** 2
        # print(loss)
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        # print(loss)
        # print(mask.shape,mask.sum())
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        # print(loss)
        return loss

    def patchify(self, imgs, data_type = None):
        """
        imgs: (N, 3, T, H, W)
        x: (N, L, patch_size**2 *3*T)
        """

        # video
        p = self.patch_size
        # print(imgs.shape)
        assert imgs.shape[3] == imgs.shape[4] and imgs.shape[3] % p == 0

        h = w = imgs.shape[3] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, imgs.shape[2], h, p, w, p))
        x = torch.einsum('ncthpwq->nhwtpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, imgs.shape[2], p**2 * 3))

        # print(x.shape) (B, N, T, patch_size**2 * 3)

        return x

    def unpatchify(self, x, data_type = None):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        # print(x.shape, 'ss')
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        t = x.shape[2]

        x = x.reshape(shape=(x.shape[0], h, w, t, p, p, 3))
        x = torch.einsum('nhwtpqc->ntchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], t, 3, h * p, h * p))

        return imgs

class TactileProbe(nn.Module):
    def __init__(self, config, num_frames, add_time_attn, tube_size, loads_from_clip=False, now_sensor='gelsight'):
        super(TactileProbe, self).__init__()

        config.vision_config.num_frames = num_frames
        config.vision_config.tube_size = tube_size

        self.load_from_clip = loads_from_clip

        if not self.load_from_clip:
            self.sensor_token = nn.Parameter(torch.zeros(10, 5, config.vision_config.hidden_size))

        self.touch_model = CLIPVisionTransformer(config.vision_config)
        # self.touch_projection = nn.Linear(config.vision_config.hidden_size, config.projection_dim, bias=False)

        if not self.load_from_clip:
            self.touch_model.embeddings.patch_embedding = nn.Conv3d(
                in_channels=config.vision_config.num_channels,
                out_channels=self.touch_model.embeddings.embed_dim,
                kernel_size=(3, self.touch_model.embeddings.patch_size, self.touch_model.embeddings.patch_size),
                stride=(3, self.touch_model.embeddings.patch_size, self.touch_model.embeddings.patch_size),
                bias=False,
            )

            if now_sensor == 'gelsight':
                self.now_sensor = torch.tensor([3]*2000, dtype=torch.int64)
            elif now_sensor=='digit':
                self.now_sensor = torch.tensor([1]*2000, dtype=torch.int64)
        
        if not self.load_from_clip:
            self.touch_projection = nn.Linear(config.vision_config.hidden_size, config.projection_dim, bias=False)

        self.touch_model.forward = self.touch_forward
        self.touch_model.embeddings.forward = self.emb_forward

    def init_head(self):
        trunc_normal_(self.head.weight, std=0.01)

    @add_start_docstrings_to_model_forward(CLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPVisionConfig)
    def touch_forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        sensor_type = None
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

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states  = self.touch_model.embeddings(pixel_values, sensor_type = sensor_type)
        hidden_states = self.touch_model.pre_layrnorm(hidden_states)

        encoder_outputs = self.touch_model.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
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

    def emb_forward(self, pixel_values: torch.FloatTensor, noise=None, sensor_type=None) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        target_dtype = self.touch_model.embeddings.patch_embedding.weight.dtype
        patch_embeds = self.touch_model.embeddings.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        pos_emb = self.touch_model.embeddings.position_embedding(self.touch_model.embeddings.position_ids)

        embeddings = patch_embeds + pos_emb[:, 1:, :]

        class_embeds = self.touch_model.embeddings.class_embedding + pos_emb[:, 0, :]
        class_embeds = class_embeds.expand(batch_size, 1, -1)

        if not self.load_from_clip:
            sensor_emb = self.sensor_token[sensor_type]
            embeddings = torch.cat([class_embeds, sensor_emb, embeddings], dim=1)
        else:
            embeddings = torch.cat([class_embeds, embeddings], dim=1)
        #embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings
    
    def forward(self, x, sensor_type = None):

        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        # if sensor_type is None:
        #     sensor_type = sensor_type.repeat(T)
            
        
        if not self.load_from_clip:
            sensor_type = self.now_sensor[:x.shape[0]].to(x.device)
            x = x.unsqueeze(1).repeat(1, 3, 1, 1, 1)
        
        # print(x.shape, sensor_type.shape)

        with torch.no_grad():
            x = self.touch_model(x, sensor_type = sensor_type)

        out = x.last_hidden_state

        _, n, d = out.shape
        out = out.view(B, -1, n, d)
        out = out.view(B, -1, d)
        
        return out
