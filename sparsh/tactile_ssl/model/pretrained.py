# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

from typing import List
import torch.nn as nn

from torchvision import models


def resnet18():
    encoder = models.resnet18(weights='ResNet18_Weights.DEFAULT')
    encoder.fc = nn.Identity()
    return encoder


def alexnet():
    encoder = models.alexnet(weights='AlexNet_Weights.DEFAULT')
    encoder.classifier = nn.Identity()
    return encoder


class AlexnetWrapper(nn.Module):
    def __init__(self):
        super(AlexnetWrapper, self).__init__()
        self.encoder = alexnet()

    def _register_hooks(self, layers: List[int]):
        self.intermediate_layers = []
        if layers is not None:
            for layer in layers:
                self.encoder.features[layer].register_forward_hook(
                    lambda model, input, output: self.intermediate_layers.append(output)
                )

    def get_intermediate_layers(self, x, layers: List[int]=None):
        self._register_hooks(layers)
        out = self.encoder(x)
        return self.intermediate_layers, out

    def forward(self, x):
        return self.encoder(x)