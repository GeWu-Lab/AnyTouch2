# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import copy
import math
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from sparsh.tactile_ssl.algorithm import Module
from sparsh.tactile_ssl.utils import apply_masks, patches_to_image, patchify_image
from sparsh.tactile_ssl.utils.ema import update_moving_average
from sparsh.tactile_ssl.utils.logging import get_pylogger, img_logger

log = get_pylogger(__name__)


class IJEPAModule(Module, nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        predictor: nn.Module,
        optim_cfg: partial,
        lr_scheduler_cfg: Optional[partial],
        wd_scheduler_cfg: Optional[partial],
        online_probes: Optional[List[nn.Module]] = None,
        online_probes_lrs: List[float] = None,
        encoder_mask_scale: Tuple[float, float] = (0.2, 0.8),
        predictor_mask_scale: Tuple[float, float] = (0.2, 0.8),
        aspect_ratio: Tuple[float, float] = (0.3, 3.0),
        num_encoder_masks: int = 1,
        num_predictor_masks: int = 4,
        min_keep_num_patches: int = 4,
        allow_mask_overlap: bool = False,
        moving_average_decay: Union[float, Tuple[float]] = 0.99,
        use_momentum=True,
        reconstruction_log_freq: int = 10,
    ):
        super().__init__()
        self.optim_partial = optim_cfg
        self.lr_scheduler_partial = lr_scheduler_cfg
        self.wd_scheduler_partial = wd_scheduler_cfg
        self.use_momentum = use_momentum
        self.encoder_mask_scale = encoder_mask_scale
        self.predictor_mask_scale = predictor_mask_scale
        self.aspect_ratio = aspect_ratio
        self.num_encoder_masks = num_encoder_masks
        self.num_predictor_masks = num_predictor_masks
        self.min_keep = min_keep_num_patches
        self.allow_mask_overlap = allow_mask_overlap
        self.log_freq_img = reconstruction_log_freq

        self.generator = torch.Generator()
        self.step = -1

        # Encoders
        self.context_encoder = encoder
        self.predictor = predictor
        self.target_encoder = copy.deepcopy(self.context_encoder)
        self.target_encoder.requires_grad_(False)

        # Momentum scheduler if moving average decay is a tuple
        self.momentum_scheduler = None
        if not isinstance(moving_average_decay, float):
            assert isinstance(moving_average_decay, list) or isinstance(
                moving_average_decay, ListConfig
            )
            assert len(moving_average_decay) == 2
            moving_average_decay = tuple(moving_average_decay)
        self.moving_average_decay = moving_average_decay

        # online probes
        self.online_probes = (
            [] if online_probes is None else nn.ModuleList(online_probes)
        )
        self.online_probes_lrs = online_probes_lrs

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def log_results(self, outputs: Dict, label: str, trainer_instance=None, step=None):
        if trainer_instance is not None:
            if trainer_instance.should_log:
                for k, v in outputs.items():
                    trainer_instance.wandb.log(
                        {
                            f"{label}/{k}": v,
                            f"global_{label}_step": step,
                        }
                    )
            if (step % self.log_freq_img == 0) and "pred_img" in outputs.keys():
                Xpred = outputs["pred_img"]
                Xorg = outputs["gt_img"] if "gt_img" in outputs.keys() else None
                img_logger(
                    wandb=trainer_instance.wandb,
                    global_step=step,
                    predictions=Xpred,
                    X=Xorg,
                    label=label,
                )

    def on_train_batch_end(self, outputs, batch, batch_idx, trainer_instance=None):
        assert (
            self.use_momentum
        ), "you do not need to update the moving average, since you have turned off momentum for the target encoder"
        assert self.target_encoder is not None, "target encoder has not been created"
        moving_average_decay = (
            next(self.momentum_scheduler)
            if self.momentum_scheduler is not None
            else self.moving_average_decay
        )
        with torch.no_grad():
            update_moving_average(
                self.target_encoder,
                self.context_encoder,
                moving_average_decay,
            )

        self.log_results(
            outputs, "train", trainer_instance, step=trainer_instance.global_step
        )

    def on_validation_batch_end(
        self, outputs: Dict, batch: Dict, batch_idx: int, trainer_instance=None
    ):
        self.log_results(
            outputs, "val", trainer_instance, step=trainer_instance.global_val_step
        )

    def _sample_block_size(self, height, width, scale, aspect_ratio_scale):
        _rand = torch.rand(1, generator=self.generator).item()
        # -- Sample block scale
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        max_keep = int(height * width * mask_scale)
        # -- Sample block aspect-ratio
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)
        # -- Compute block height and width (given scale and aspect-ratio)
        h = int(round(math.sqrt(max_keep * aspect_ratio)))
        w = int(round(math.sqrt(max_keep / aspect_ratio)))
        while h > height:
            h -= 1
        while w > width:
            w -= 1

        return (h, w)

    def _sample_block_mask(self, height, width, b_size, acceptable_regions=None):
        h, w = b_size

        def constrain_mask(mask, tries=0):
            """Helper to restrict given mask to a set of acceptable regions"""
            N = max(int(len(acceptable_regions) - tries), 0)
            for k in range(N):
                mask *= acceptable_regions[k]

        tries = 0
        timeout = og_timeout = 20
        valid_mask = False
        while not valid_mask:
            # -- Sample block top-left corner
            top = torch.randint(0, height - h + 1, (1,), generator=self.generator)
            left = torch.randint(0, width - w + 1, (1,), generator=self.generator)
            mask = torch.zeros((height, width), dtype=torch.int32)
            mask_complement = torch.ones_like(mask)
            mask[top : top + h, left : left + w] = 1
            # -- Constrain mask to a set of acceptable regions
            if acceptable_regions is not None:
                constrain_mask(mask, tries)
            mask = torch.nonzero(mask.flatten())
            # -- If mask too small try again
            valid_mask = len(mask) > self.min_keep
            if not valid_mask:
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = og_timeout
                    log.warning(
                        f'Mask generator says: "Valid mask not found, decreasing acceptable-regions [{tries}]"'
                    )
        mask = mask.squeeze()
        # --
        mask_complement = torch.ones((height, width), dtype=torch.int32)
        mask_complement[top : top + h, left : left + w] = 0
        # --
        return mask, mask_complement

    def get_masks_img(self, masks):
        p = self.context_encoder.patch_size
        im_size = self.context_encoder.img_size
        num_patches = self.target_encoder.patch_embed.num_patches
        d = p**2 * 3
        b = masks.shape[0]
        mask_bin = torch.zeros((b, num_patches))
        for i in range(b):
            mask_bin[i, masks[i]] = 1

        mask_bin = mask_bin.unsqueeze(-1).repeat(1, 1, d)
        mask_bin = patches_to_image(mask_bin, p, im_size)
        mask_bin = mask_bin.detach().permute(0, 2, 3, 1).cpu().numpy()

        return mask_bin

    def plot_masks(self, encoder_masks, predictor_masks, x):
        import matplotlib.pyplot as plt

        mask_enc_bin = self.get_masks_img(encoder_masks[0])
        _, axs = plt.subplots(5, 10, figsize=(20, 20))
        for id, ax in enumerate(axs.flatten()):
            encoder_mask = (
                x[id, 0:3, :, :].permute(1, 2, 0).detach().cpu().numpy()
                * mask_enc_bin[id]
            )
            ax.imshow(encoder_mask)
            ax.set_axis_off()
        plt.show()

        id = 0
        _, axs = plt.subplots(1, 4, figsize=(20, 20))
        for i in range(len(predictor_masks)):
            mask_pred_bin = self.get_masks_img(predictor_masks[i])
            predictor_mask = (
                x[id, 0:3, :, :].permute(1, 2, 0).detach().cpu().numpy()
                * mask_pred_bin[id]
            )
            axs[i].imshow(predictor_mask)
            axs[i].set_title(f"Predictor/Target Mask {i}")
            axs[i].set_axis_off()
        plt.show()

    def sample_masks(self, x):
        batch_size, _, image_height, image_width = x.shape
        height, width = (
            image_height // self.context_encoder.patch_size,
            image_width // self.context_encoder.patch_size,
        )
        predictor_size = self._sample_block_size(
            height, width, self.predictor_mask_scale, self.aspect_ratio
        )
        encoder_size = self._sample_block_size(
            height, width, self.encoder_mask_scale, (1.0, 1.0)
        )
        collated_predictor_masks, collated_encoder_masks = [], []
        min_keep_encoder_patches, min_keep_predictor_patches = (
            height * width,
            height * width,
        )
        for _ in range(batch_size):
            masks_predictor, masks_complement = [], []
            for _ in range(self.num_predictor_masks):
                mask, mask_complement = self._sample_block_mask(
                    height, width, predictor_size
                )
                masks_predictor.append(mask)
                masks_complement.append(mask_complement)
                min_keep_predictor_patches = min(min_keep_predictor_patches, len(mask))
            collated_predictor_masks.append(masks_predictor)

            acceptable_regions = masks_complement

            if self.allow_mask_overlap:
                acceptable_regions = None

            masks_encoder = []
            for _ in range(self.num_encoder_masks):
                mask, _ = self._sample_block_mask(
                    height, width, encoder_size, acceptable_regions
                )
                masks_encoder.append(mask)
                min_keep_encoder_patches = min(min_keep_encoder_patches, len(mask))
            collated_encoder_masks.append(masks_encoder)

        collated_encoder_masks = [
            [cm[:min_keep_encoder_patches] for cm in masks]
            for masks in collated_encoder_masks
        ]
        collated_predictor_masks = [
            [cm[:min_keep_predictor_patches] for cm in masks]
            for masks in collated_predictor_masks
        ]
        predictor_masks = torch.utils.data.default_collate(collated_predictor_masks)
        encoder_masks = torch.utils.data.default_collate(collated_encoder_masks)
        for i in range(len(encoder_masks)):
            encoder_masks[i] = encoder_masks[i].to(x.device)
        for i in range(len(predictor_masks)):
            predictor_masks[i] = predictor_masks[i].to(x.device)

        return encoder_masks, predictor_masks

    def forward(
        self,
        x: torch.Tensor,
        encoder_masks: List[torch.Tensor] = None,
        predictor_masks: List[torch.Tensor] = None,
        return_embedding=False,
    ):
        if return_embedding:
            return self.online_encoder(x)
        assert (
            encoder_masks is not None and predictor_masks is not None
        ), "Masks are required for JEPAModule during training"

        # self.plot_masks(encoder_masks, predictor_masks, x)

        # Raise to make sure context encoder implements taking masks as an argument
        context_features = self.context_encoder(x, encoder_masks)
        context_prediction = self.predictor(
            context_features, encoder_masks, predictor_masks
        )
        with torch.no_grad():
            target_features = self.target_encoder(x)
            target_features = F.layer_norm(target_features, [target_features.shape[-1]])
            masked_target_features_x = apply_masks(target_features, predictor_masks)
            masked_target_features = einops.repeat(
                masked_target_features_x,
                "(p b) n d -> (p k b) n d",
                k=len(encoder_masks),
                p=len(predictor_masks),
            )
        loss = F.smooth_l1_loss(context_prediction, masked_target_features.detach())
        return loss.mean()

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        self.step = self.step + 1
        self.generator.manual_seed(self.step)

        x = batch["image"]

        encoder_masks, predictor_masks = self.sample_masks(x)
        loss = self.forward(x, encoder_masks, predictor_masks)

        output = {
            "ssl_loss": loss.item(),
        }

        # online probes
        with torch.no_grad():
            embedding = self.target_encoder(x)
            embedding = F.layer_norm(embedding, (embedding.size(-1),))
            x_patch_gt = patchify_image(x, self.context_encoder.patch_size)

        online_probes_loss = 0.0
        pred_img = None
        # TODO: Currently only reconstruction probe is supported. Missing targets for other probes
        for probe in self.online_probes:
            probe_name = probe.probe_name
            if probe_name == "reconstruction":
                probe_loss, decoded_x = probe(
                    embedding, target=x_patch_gt, img_shape=x.shape
                )
                online_probes_loss += probe_loss
                output[f"{probe_name}_loss"] = probe_loss.item()
                pred_img = patches_to_image(
                    decoded_x.detach(),
                    self.context_encoder.patch_size,
                    self.context_encoder.img_size,
                ).float()
                output["pred_img"] = pred_img[:, 0:3, :, :].unsqueeze(2)
                output["gt_img"] = x[:, 0:3, :, :].unsqueeze(2)
            else:
                raise NotImplementedError(f"Probe {probe_name} missing target")

        loss += online_probes_loss

        output["loss"] = loss  # type: ignore
        output["online_probes_loss"] = online_probes_loss

        return output

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        return self.training_step(batch, batch_idx)

    def configure_optimizers(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, num_iterations_per_epoch, num_epochs
    ) -> Tuple[torch.optim.Optimizer, Optional[Dict], Optional[Dict]]:
        param_dict = {
            pn: p
            for pn, p in self.named_parameters()
            if not pn.startswith("online_probes")
        }
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]

        optim_groups = [
            {"params": decay_params},
            {"params": nodecay_params, "WD_exclude": True, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        for probe, lr in zip(self.online_probes, self.online_probes_lrs):
            trainable_probe_params = {
                pn: p for pn, p in probe.named_parameters() if p.requires_grad
            }
            optim_groups.append({"params": trainable_probe_params.values(), "lr": lr})

        log.info(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        log.info(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )

        optimizer = self.optim_partial(optim_groups)
        if self.lr_scheduler_partial is None:
            return optimizer, None, None

        lr_scheduler = self.lr_scheduler_partial(
            optimizer=optimizer,
            T_max=int(num_epochs * num_iterations_per_epoch),
            steps_per_epoch=num_iterations_per_epoch,
        )
        if isinstance(self.moving_average_decay, tuple):
            self.momentum_scheduler = (
                self.moving_average_decay[0]
                + i
                * (self.moving_average_decay[1] - self.moving_average_decay[0])
                / (num_epochs * num_iterations_per_epoch)
                for i in range(int(num_epochs * num_iterations_per_epoch) + 1)
            )
        if self.wd_scheduler_partial is None:
            return (
                optimizer,
                {
                    "scheduler": lr_scheduler,
                    "interval": "step",
                    "monitor": None,
                },
                None,
            )

        wd_scheduler = self.wd_scheduler_partial(
            optimizer,
            T_max=int(num_epochs * num_iterations_per_epoch),
        )
        return (
            optimizer,
            {"scheduler": lr_scheduler, "interval": "step", "monitor": None},
            {"wd_scheduler": wd_scheduler, "interval": "step", "frequency": 1},
        )
