# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from sparsh.tactile_ssl.algorithm.module import Module
from sparsh.tactile_ssl.utils.logging import get_pylogger

log = get_pylogger(__name__)


class SLModule(Module, nn.Module):
    def __init__(
        self,
        model_encoder: nn.Module,
        model_task: nn.Module,
        optim_cfg: partial,
        scheduler_cfg: Optional[partial],
        checkpoint_encoder: Optional[str] = None,
        checkpoint_task: Optional[str] = None,
        train_encoder: bool = False,
        encoder_type: str = "jepa",
        is_anytouch: bool = False,
    ) -> None:
        super().__init__()
        self.model_task: nn.Module = model_task
        self.model_encoder: nn.Module = model_encoder
        self.train_encoder: bool = train_encoder
        self.encoder_type: str = encoder_type

        if checkpoint_encoder is not None:
            log.info("Loading encoder ONLY from checkpoint.")
            self.load_encoder(checkpoint_encoder)
        else:
            log.info("No checkpoint provided. Training from scratch.")
        
        if checkpoint_task is not None:
            log.info("Loading task decoder from checkpoint.")
            self.load_task(checkpoint_task)

        # freeze encoder
        if not self.train_encoder:
            self.model_encoder.requires_grad_(False)
            self.model_encoder.eval()
        self.scheduler_partial = scheduler_cfg
        self.optim_partial = optim_cfg

    def load_task(self, checkpoint_task: str):
        try:
            state_dict = torch.load(checkpoint_task)
            # check if there are keys starting with "model_encoder."
            if any([key.startswith("model_encoder.") for key in state_dict.keys()]):
                log.info("Found encoder in task checkpoint. Loading encoder and decoder from task checkpoint.")
                self.load_state_dict(state_dict, strict=False)
                return
            else:
                state_dict = {
                    key.replace("model_task.", ""): value
                    for key, value in state_dict.items()
                }
                self.model_task.load_state_dict(state_dict, strict=False)
                log.info(f"Loaded task model from {checkpoint_task}")
        except:
            # add to state_dict_light only keys that start with model_encoder
            try:
                state_dict_light = {
                    key.replace("model_encoder.", ""): value
                    for key, value in torch.load(checkpoint_task).items()
                    if key.startswith("model_encoder.")
                }
                self.model_encoder.load_state_dict(state_dict_light, strict=False)
                log.info(f"Loaded encoder from {checkpoint_task}")
            except:
                log.info(f"Could not load task model from {checkpoint_task}")

    def load_encoder(self, checkpoint_encoder: str):
        log.info(f"Loading encoder from {checkpoint_encoder}")
        checkpoint = torch.load(checkpoint_encoder)
        if "jepa" in self.encoder_type:
            encoder_key = "target_encoder"
        elif "dino" in self.encoder_type:
            encoder_key = "teacher_encoder.backbone"
        else:
            encoder_key = "encoder"
        # get the keys in the checkpoint that contain the encoder
        target_keys = [key for key in checkpoint["model"].keys() if encoder_key in key]
        if 'backbone' in target_keys[0] and 'backbone' not in encoder_key:
            encoder_key = encoder_key + '.backbone'
        # remove the prefix from the keys
        new_keys = [key.replace(f"{encoder_key}.", "") for key in target_keys]
        # create a state_dict  with keys target_keys from the checkpoint
        new_state_dict = {
            new_key: checkpoint["model"][target_key]
            for new_key, target_key in zip(new_keys, target_keys)
        }
        # load the state_dict into the model
        self.model_encoder.load_state_dict(new_state_dict, strict=False)
        log.info(f"Loaded encoder from {checkpoint_encoder}")
    
    def forward(self, x, *args, **kwargs):  # noqa
        raise NotImplementedError

    def training_step(self, batch: Dict[str, Any], batch_idx: int, *args, **kwargs):  # noqa
        raise NotImplementedError

    def validation_step(self, *args, **kwargs):  # noqa
        raise NotImplementedError

    def test_step(self, *args, **kwargs):  # noqa
        raise NotImplementedError

    def configure_optimizers(
        self, num_iterations_per_epoch, num_epochs, *args, **kwargs
    ) -> Tuple[torch.optim.Optimizer, Optional[Dict], Optional[Dict]]:
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]

        optim_groups = [
            {"params": decay_params},
            {"params": nodecay_params, "WD_exclude": True, "weight_decay": 0.0},
        ]
        optimizer = self.optim_partial(optim_groups)


        if self.scheduler_partial is not None:
            lr_scheduler = self.scheduler_partial(
                optimizer=optimizer,
                T_max=int(1.0 * num_epochs * num_iterations_per_epoch),
                warmup_steps=int(num_iterations_per_epoch),
            )
            return (
                optimizer,
                {
                    "scheduler": lr_scheduler,
                    "interval": "step",
                    "monitor": None,
                },
                None,
            )

        return optimizer, None, None
