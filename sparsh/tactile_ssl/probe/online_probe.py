# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from sparsh.tactile_ssl.utils.logging import get_pylogger

log = get_pylogger(__name__)


class OnlineProbeModule(nn.Module):
    def __init__(
        self,
        probe_name: str,
        decoder: nn.Module,
        loss_fn: partial,
    ):
        super().__init__()
        self.probe_name = probe_name
        self.decoder = decoder
        self.loss_fn = loss_fn

    def forward_decoder(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.decoder(z, **kwargs)
        return x

    def forward(self, embedding: torch.Tensor, target: torch.Tensor, **kwargs) -> Dict:
        prediction = self.forward_decoder(embedding, **kwargs)
        loss = self.loss_fn(prediction, target)
        return loss, prediction
