# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode, align_corners=False):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )
        return x


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU"""

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input"""

    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


def upsample(x):
    """Upsample input tensor by a factor of 2"""
    return F.interpolate(x, scale_factor=2, mode="nearest")


class NormalShearHead(nn.Module):
    def __init__(self, features, use_skips=True):
        super(NormalShearHead, self).__init__()
        self.use_skips = use_skips
        # num_output_channels = 3
        self.scale_flow = 20.0
        num_ch_out = 128
        num_ch_in = features
        self.upconv_0 = ConvBlock(num_ch_in, num_ch_out)
        self.upconv_1 = ConvBlock(num_ch_in + num_ch_out, num_ch_out)
        self.dispconv = Conv3x3(num_ch_out, 1)
        self.sigmoid = nn.Sigmoid()

        self.shear_mlp = nn.Sequential(
            Conv3x3(num_ch_out, num_ch_out // 2),
            nn.GELU(),
            Conv3x3(num_ch_out // 2, 2),
            nn.Tanh(),
        )

    def forward(self, input_features, mode="normal_shear"):
        x = input_features
        x = self.upconv_0(x)
        x = [x]
        if self.use_skips:
            x += [input_features]
        x = torch.cat(x, 1)
        x = self.upconv_1(x)
        x = upsample(x)

        if mode == "normal_shear":
            
            # normal field. Apply sigmoid activation to get values between 0 and 1
            disp = self.sigmoid(self.dispconv(x))

            # shear field. Has a tanh activation to get values between -1 and 1
            # flow field is scaled by scale_flow (20) to make it more visible
            shear = self.shear_mlp(x) * self.scale_flow

            forces = torch.cat([disp, shear], dim=1)
        elif mode == "normal":
            forces = self.sigmoid(self.dispconv(x))
        elif mode == "shear":
            forces = self.shear_mlp(x) * self.scale_flow
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        return forces
