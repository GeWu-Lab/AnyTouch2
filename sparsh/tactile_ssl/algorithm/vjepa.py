# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import copy
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import einops
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import ListConfig

from sparsh.tactile_ssl.algorithm import Module
from sparsh.tactile_ssl.utils import (
    AverageMeter,
    apply_masks,
    patches_to_image,
    patchify_image,
)
from sparsh.tactile_ssl.utils.ema import update_moving_average
from sparsh.tactile_ssl.utils.logging import get_pylogger, img_logger
from sparsh.tactile_ssl.utils.masking import (
    MaskCollator,
    MultiMaskWrapper,
    PredictorMultiMaskWrapper,
)

log = get_pylogger(__name__)


class VJEPAModule(Module, nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        predictor: nn.Module,
        optim_cfg: partial,
        lr_scheduler_cfg: Optional[partial],
        wd_scheduler_cfg: Optional[partial],
        moving_average_decay: Union[float, Tuple[float, ...]] = [0.998, 1.0],
        img_size: int = 224,
        patch_size: int = 16,
        num_frames: int = 16,
        tubelet_size: int = 2,
        mask_cfg: Optional[Dict[str, Any]] = None,
        loss_cfg: Optional[Dict[str, Any]] = None,
        use_momentum: bool = True,
        online_probes: Optional[List[nn.Module]] = None,
        online_probes_lrs: List[float] = None,
        reconstruction_log_freq: int = 10,
    ):
        super().__init__()
        self.optim_partial = optim_cfg
        self.lr_scheduler_partial = lr_scheduler_cfg
        self.wd_scheduler_partial = wd_scheduler_cfg
        self.mask_cfg = mask_cfg
        self.loss_cfg = loss_cfg
        self.moving_average_decay = moving_average_decay
        self.momentum_scheduler = None
        self.step = -1
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.num_frames = num_frames
        self.img_size = img_size
        self.use_momentum = use_momentum
        self.log_freq_img = reconstruction_log_freq
        self.tmp_stride = list(range(0, self.num_frames, self.tubelet_size))

        # context encoder and predictor
        self.context_encoder = encoder
        self.predictor = predictor
        self.context_encoder = MultiMaskWrapper(self.context_encoder)
        self.predictor = PredictorMultiMaskWrapper(self.predictor)
        for m in encoder.modules():
            self._init_weights(m)

        for m in predictor.modules():
            self._init_weights(m)

        log.info(encoder)
        log.info(predictor)

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        log.info(f"Encoder number of parameters: {count_parameters(encoder)}")
        log.info(f"Predictor number of parameters: {count_parameters(predictor)}")

        # target encoder
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # mask
        self.mask_collator = MaskCollator(
            cfgs_mask=mask_cfg,
            crop_size=img_size,
            num_frames=num_frames,
            patch_size=(patch_size, patch_size),
            tubelet_size=tubelet_size,
        )
        self.num_masks = len(mask_cfg)

        # meters
        self._create_meters()
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

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    def _create_meters(self):
        self.jepa_loss_meter = AverageMeter()
        self.reg_loss_meter = AverageMeter()
        self.mask_meters = [AverageMeter() for _ in range(self.num_masks)]

    def _reset_meters(self):
        self.jepa_loss_meter.reset()
        self.reg_loss_meter.reset()
        for m in self.mask_meters:
            m.reset()

    def on_train_epoch_start(self):
        self._reset_meters()

    def update_target_encoder(self):
        assert self.momentum_scheduler is not None, "Momentum scheduler not set"
        with torch.no_grad():
            m = next(self.momentum_scheduler)
            m = m**self.k
            for param_q, param_k in zip(
                self.context_encoder.parameters(), self.target_encoder.parameters()
            ):
                param_k.data.mul_(m).add_((1.0 - m) * param_q.detach().data)

    def _forward_target(self, x, predictor_masks):
        with torch.no_grad():
            h = self.target_encoder(x)
            h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim  [B, N, D]
            # -- create targets (masked regions of h)
            h = apply_masks(h, predictor_masks, concat=False)
            return h

    def _forward_context(self, x, encoder_masks, predictor_masks):
        """
        Returns list of tensors of shape [B, N, D], one for each
        mask-pred.
        """
        z = self.context_encoder(x, encoder_masks)
        z = self.predictor(z, encoder_masks, predictor_masks)
        return z

    def loss_fn(self, z, h, predictor_masks):
        loss = 0.0
        # Compute loss and accumulate for each mask-enc/mask-pred pair
        for zi, hi in zip(z, h):
            loss += (
                torch.mean(torch.abs(zi - hi) ** self.loss_cfg.loss_exp)
                / self.loss_cfg.loss_exp
            )
        loss /= len(predictor_masks)
        return loss

    def reg_fn(self, z):
        return sum([torch.sqrt(zi.var(dim=1) + 0.0001) for zi in z]) / len(z)

    def patchify_img(self, img):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        _, _, T, H, W = img.shape

        tubelet_size = self.tubelet_size
        N_t = T // tubelet_size
        N_h = H // self.patch_size
        N_w = W // self.patch_size

        x = img.reshape(
            shape=(
                img.shape[0],
                3,
                N_t,
                tubelet_size,
                N_h,
                self.patch_size,
                N_w,
                self.patch_size,
            )
        )
        x = torch.einsum("nctkhpwq->nthwpqc", x)
        x = x.reshape(shape=(x.shape[0], N_h * N_w * N_t, self.patch_size**2 * 3))
        return x

    def patch2img(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """

        N_t = self.num_frames // self.tubelet_size
        N_h = self.img_size[0] // self.patch_size
        N_w = self.img_size[1] // self.patch_size

        x = x.reshape(
            shape=(x.shape[0], N_t, N_h, N_w, self.patch_size, self.patch_size, 3)
        )
        x = torch.einsum("nthwpqc->ncthpwq", x)
        imgs = x.reshape(
            shape=(x.shape[0], 3, N_t, N_h * self.patch_size, N_w * self.patch_size)
        )

        return imgs

    def forward(
        self,
        x: torch.Tensor,
        encoder_masks: List[torch.Tensor] = None,
        predictor_masks: List[torch.Tensor] = None,
    ):
        loss_jepa, loss_reg = 0.0, 0.0
        target_prediction = self._forward_target(x, predictor_masks)
        context_prediction = self._forward_context(x, encoder_masks, predictor_masks)
        # context_prediction = target_prediction
        loss_jepa = self.loss_fn(
            context_prediction, target_prediction, predictor_masks
        )  # jepa prediction loss
        pstd_z = self.reg_fn(context_prediction)  # predictor variance across patches
        loss_reg += torch.mean(F.relu(1.0 - pstd_z))
        loss = loss_jepa + self.loss_cfg.reg_coeff * loss_reg

        output = {
            "loss": loss,
            "loss_jepa+reg": loss.item(),
            "loss_jepa": loss_jepa.item(),
            "loss_jepa_avg": self.jepa_loss_meter.avg,
            "loss_reg": loss_reg.item(),
            "loss_reg_avg": self.reg_loss_meter.avg,
            "pred_img": None,
            "gt_img": None,
            "online_probes_loss": None,
        }

        # online probes
        if len(self.online_probes) == 0:
            return loss, loss_jepa, loss_reg, None, None

        with torch.no_grad():
            h = self.target_encoder(x)
            h = F.layer_norm(h, (h.size(-1),))
            x_patch_gt = self.patchify_img(x)

        online_probes_loss = 0.0
        pred_img = None
        gt_img = x[:, :, self.tmp_stride, :, :]

        for probe in self.online_probes:
            probe_name = probe.probe_name
            if probe_name == "reconstruction":
                probe_loss, decoded_x = probe(h, target=x_patch_gt, img_shape=x.shape)
                output[f"{probe_name}_loss"] = probe_loss.item()
                online_probes_loss += probe_loss
                pred_img = self.patch2img(decoded_x.detach()).float()
                output[f"pred_img"] = pred_img
                output[f"gt_img"] = gt_img
            else:
                raise NotImplementedError(f"Probe {probe_name} missing target")

        loss += online_probes_loss

        output["loss"] = loss
        output["online_probes_loss"] = online_probes_loss

        return output

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

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        self.step = self.step + 1

        x = batch["image"]
        # self.show_clip(x[0])

        x = einops.rearrange(x, "b t c h w -> b c t h w")
        encoder_masks, predictor_masks = self.mask_collator(x)
        encoder_masks = [mask.to(x.device) for mask in encoder_masks]
        predictor_masks = [mask.to(x.device) for mask in predictor_masks]

        # forward model
        # loss, loss_jepa, loss_reg, probes_loss, probes_img, gt_img = self.forward(
        #     x, encoder_masks, predictor_masks
        # )
        output = self.forward(x, encoder_masks, predictor_masks)

        # Update meters
        self.jepa_loss_meter.update(output["loss_jepa"])
        self.reg_loss_meter.update(output["loss_reg"])
        for _i, m in enumerate(self.mask_meters):
            m.update(encoder_masks[_i][0].size(-1))

        output["loss_jepa_avg"] = self.jepa_loss_meter.avg
        output["loss_reg_avg"] = self.reg_loss_meter.avg

        for i, m in enumerate(self.mask_meters):
            output[f"mask_{i}"] = m.avg

        return output

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        return self.training_step(batch, batch_idx)

    def configure_optimizers(
        self,
        num_iterations_per_epoch,
        num_epochs,
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
