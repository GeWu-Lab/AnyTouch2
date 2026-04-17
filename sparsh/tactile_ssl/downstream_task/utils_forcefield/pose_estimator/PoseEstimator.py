# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from .resnet_encoder import ResnetEncoder
from .pose_decoder import PoseDecoder
from .utils import transformation_from_parameters


class PoseEstimator(nn.Module):
    def __init__(
        self,
        num_encoder_layers,
        frame_ids=[0, -1],
    ):
        super().__init__()
        self.frame_ids = frame_ids
        self.encoder = ResnetEncoder(
            num_layers=num_encoder_layers, pretrained=False, num_input_images=2
        )
        self.decoder = PoseDecoder(
            num_ch_enc=self.encoder.num_ch_enc,
            num_input_features=1,
            num_frames_to_predict_for=2,
        )

    def forward(self, inputs):
        """Predict poses between input frames for monocular sequences."""
        outputs = {}

        # In this setting, we compute the pose to each source frame via a
        # separate forward pass through the pose network.
        pose_feats = {0: inputs[:, 0:3, :, :], -1: inputs[:, 3:6, :, :]}

        for f_i in self.frame_ids[1:]:
            # To maintain ordering we always pass frames in temporal order
            if f_i < 0:
                pose_inputs = [pose_feats[f_i], pose_feats[0]]
            else:
                pose_inputs = [pose_feats[0], pose_feats[f_i]]

            pose_inputs = [self.encoder(torch.cat(pose_inputs, 1))]
            axisangle, translation = self.decoder(pose_inputs)
            outputs[("axisangle", f_i)] = axisangle
            outputs[("translation", f_i)] = translation

            outputs[("cam_T_cam", f_i)] = transformation_from_parameters(
                axisangle[:, 0], translation[:, 0], invert=(f_i < 0)
            )

        return outputs
