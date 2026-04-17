# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from sparsh.tactile_ssl.algorithm.module import Module
from sparsh.tactile_ssl.model.vision_transformer import VisionTransformer
from sparsh.tactile_ssl.utils.logging import get_pylogger, img_logger
from sparsh.tactile_ssl.utils import patchify_image, patches_to_image

log = get_pylogger(__name__)


class MAEModule(Module, nn.Module):
    def __init__(
        self,
        encoder: VisionTransformer,
        decoder: partial,
        optim_cfg: partial,
        lr_scheduler_cfg: Optional[partial],
        wd_scheduler_cfg: Optional[partial],
        mask_type: str = "random",
        mask_ratio: float = 0.75,
        use_momentum=True,
        norm_pix_loss: bool = False,
        log_freq_reconstruction: int = 1000,
    ):
        super().__init__()
        self.optim_partial = optim_cfg
        self.lr_scheduler_partial = lr_scheduler_cfg
        self.wd_scheduler_partial = wd_scheduler_cfg
        self.use_momentum = use_momentum
        self.norm_pix_loss = norm_pix_loss
        self.log_freq_img = log_freq_reconstruction

        self.mask_type = mask_type
        assert mask_type in [
            "random",
            "block",
        ], f"mask_type={mask_type} not supported. Must be 'random' or 'block'"
        self.mask_ratio: float = mask_ratio

        if mask_type == "random":
            self.mask_fn = self.random_masking
        else:
            raise NotImplementedError("Implement compatibility with IJEPA masking")

        self.num_encoder_masks = 1

        # MAE Encoder
        self.encoder: VisionTransformer = encoder
        assert hasattr(
            self.encoder, "patch_embed"
        ), "Encoder must have patch_embed module"
        self.in_chans = self.encoder.patch_embed.in_chans
        if isinstance(self.encoder.patch_embed.patch_size, tuple):
            self.patch_size = self.encoder.patch_embed.patch_size[0]
        else:
            self.patch_size = self.encoder.patch_embed.patch_size
        self.num_patches = self.encoder.patch_embed.num_patches
        self.embed_dim = self.encoder.embed_dim
        self.img_size = self.encoder.img_size
        self.in_chans = self.encoder.in_chans

        # MAE Decoder
        self.decoder = self.load_decoder(decoder)

    def load_decoder(self, decoder_partial: partial) -> nn.Module:
        decoder = decoder_partial(
            in_chans=self.in_chans,
            img_size=self.encoder.img_size,
            input_embed_dim=self.embed_dim,
            patch_size=self.patch_size,
        )
        return decoder

    def random_masking(self, x):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        From: https://github.com/facebookresearch/mae/tree/main
        """
        # N, L, D = x.shape  # batch, length, dim
        N = x.shape[0]
        L = self.num_patches
        len_keep = int(L * (1 - self.mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return (
            ids_keep,
            mask,
            ids_restore,
        )

    def forward_encoder(
        self, x: torch.Tensor, ids_mask_visible: Optional[torch.Tensor] = None
    ):
        embedding = self.encoder(x, [ids_mask_visible])
        return embedding

    def forward(self, x: torch.Tensor):
        ids_mask_visible, mask, ids_restore_mask = self.mask_fn(x)
        latent = self.forward_encoder(x, ids_mask_visible)
        x_pred = self.decoder(latent, x.shape, ids_restore_mask)
        return x_pred, mask

    def compute_loss(self, imgs, pred, mask):
        """
        imgs: [N, C, H, W]
        mask: [N, L], 0 is keep, 1 is remove,
        From: https://github.com/facebookresearch/mae/tree/main
        """
        target = patchify_image(imgs, self.patch_size)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        x = batch["image"]
        x_pred, mask_img = self.forward(x)
        loss = self.compute_loss(x, x_pred, mask_img)

        imgs_pred = patches_to_image(x_pred.detach(), self.patch_size, self.img_size)
        imgs_pred = imgs_pred[:, 0:3, :, :]
        imgs_gt = x[:, 0:3, :, :]
        output = {"loss": loss, "pred_img": imgs_pred, "gt_img": imgs_gt}
        return output

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        return self.training_step(batch, batch_idx)

    def configure_optimizers(
        self, num_iterations_per_epoch, num_epochs
    ) -> Tuple[torch.optim.Optimizer, Optional[Dict], Optional[Dict]]:
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params},
            {"params": nodecay_params, "WD_exclude": True, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        log.info(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        log.info(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )

        optimizer = self.optim_partial(optim_groups, betas=(0.9, 0.95))

        if self.lr_scheduler_partial is None:
            return optimizer, None, None
        lr_scheduler = self.lr_scheduler_partial(
            optimizer=optimizer,
            T_max=int(num_epochs * num_iterations_per_epoch),
            steps_per_epoch=num_iterations_per_epoch,
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

    def log_results(
        self,
        outputs: Dict,
        label: str,
        trainer_instance=None,
        step: Optional[int] = None,
    ):
        if step is not None:
            if trainer_instance is not None:
                if trainer_instance.should_log:
                    trainer_instance.wandb.log(
                        {
                            f"{label}/loss": outputs["loss"],
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
        if trainer_instance is None:
            return
        self.log_results(
            outputs,
            "train",
            step=trainer_instance.global_step,
            trainer_instance=trainer_instance,
        )

    def on_validation_batch_end(self, outputs, batch, batch_idx, trainer_instance=None):
        if trainer_instance is None:
            return
        self.log_results(
            outputs,
            "val",
            step=trainer_instance.global_val_step,
            trainer_instance=trainer_instance,
        )
