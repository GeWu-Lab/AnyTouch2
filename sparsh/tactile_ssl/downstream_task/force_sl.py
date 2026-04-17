# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Tuple, Optional, List, Union
from matplotlib.axes import SubplotBase
from numpy.typing import NDArray
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sparsh.tactile_ssl.utils.logging import get_pylogger
from sparsh.tactile_ssl.downstream_task.sl_module import SLModule
from sparsh.tactile_ssl.downstream_task.attentive_pooler import AttentivePooler
from sparsh.tactile_ssl.utils.plotting_forces import plot_correlation, plot_forces_error
from sparsh.tactile_ssl.model import VIT_EMBED_DIMS


log = get_pylogger(__name__)

class ForceLinearProbe(nn.Module):
    def __init__(
        self,
        embed_dim='base',
        num_heads=12,
        mlp_ratio=4.0,
        depth=1,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        qkv_bias=True,
        complete_block=True,
        with_last_activations=False,
        is_anytouch=False,
    ):
        super().__init__()
        if is_anytouch:
            embed_dim = VIT_EMBED_DIMS[f"anytouch2_{embed_dim}"]
            if embed_dim == 512:
                num_heads = 8
        else:
            embed_dim = VIT_EMBED_DIMS[f"vit_{embed_dim}"]
        self.pooler = AttentivePooler(
            num_queries=1,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            depth=depth,
            norm_layer=norm_layer,
            init_std=init_std,
            qkv_bias=qkv_bias,
            complete_block=complete_block,
        )
        self.probe = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, 3),
        )
        self.with_last_activations = with_last_activations

    def forward(self, x):
        # print('222', x.shape)
        x = self.pooler(x).squeeze(1)
        x = self.probe(x)
        if self.with_last_activations:
            x[:, -1] = F.sigmoid(x[:, -1])
            x[:, 0:2] = F.tanh(x[:, 0:2])
        return x


class ForceSLModule(SLModule):
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
    ):
        super().__init__(
            model_encoder=model_encoder,
            model_task=model_task,
            optim_cfg=optim_cfg,
            scheduler_cfg=scheduler_cfg,
            checkpoint_encoder=checkpoint_encoder,
            checkpoint_task=checkpoint_task,
            train_encoder=train_encoder,
            encoder_type=encoder_type,
            is_anytouch=is_anytouch,
        )
        self.val_pred = []
        self.val_gt = []
        self.val_force_scale = []
        self.min_force_rmse = [float("inf"), float("inf"), float("inf")]
        self.min_force_loss = float("inf")

    def forward(self, x: torch.Tensor):
        z = self.model_encoder(x)
        # print('111', z.shape)
        if self.train_encoder:
            y_pred = self.model_task(z)
        else:
            y_pred = self.model_task(z.detach())
        return y_pred

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        x = batch["image"]
        y_gt = batch["force"]
        force_scale = batch["force_scale"]

        y_pred = self.forward(x)
        
        # print(x.shape, y_gt.shape, force_scale.shape, y_pred.shape) torch.Size([512, 4, 3, 224, 224]) torch.Size([512, 3]) torch.Size([512, 3]) torch.Size([512, 3])
        # exit(0)
        # adding beta to have bias towards the quadratic error 
        loss = F.smooth_l1_loss(y_pred, y_gt, beta=0.02)

        y_pred = y_pred.detach() 
        y_pred_scaled = y_pred * force_scale
        y_gt_scaled = y_gt.detach() * force_scale
        mse_xyz = F.mse_loss(y_pred_scaled, y_gt_scaled, reduction="none").mean(dim=0)
        return {
            "loss": loss,
            "rmse_xyz": torch.sqrt(mse_xyz),
            "y_pred": y_pred,
        }

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        return self.training_step(batch, batch_idx)

    def log_metrics(self, outputs, step, trainer_instance=None, label="train"):
        if trainer_instance is not None:
            trainer_instance.wandb.log(
                {
                    f"{label}/loss": outputs["loss"],
                    f"global_{label}_step": step,
                }
            )
            trainer_instance.wandb.log(
                {
                    f"{label}/rmse_Fx": outputs["rmse_xyz"][0],
                    f"global_{label}_step": step,
                }
            )
            trainer_instance.wandb.log(
                {
                    f"{label}/rmse_Fy": outputs["rmse_xyz"][1],
                    f"global_{label}_step": step,
                }
            )
            trainer_instance.wandb.log(
                {
                    f"{label}/rmse_Fz": outputs["rmse_xyz"][2],
                    f"global_{label}_step": step,
                }
            )

    def on_train_batch_end(self, outputs, batch, batch_idx, trainer_instance=None):
        self.log_metrics(outputs, trainer_instance.global_step, trainer_instance)

    def on_validation_batch_end(
        self, outputs: Dict, batch: Dict, batch_idx: int, trainer_instance=None
    ):
        self.val_pred.append(outputs["y_pred"])
        self.val_gt.append(batch["force"])
        self.val_force_scale.append(batch["force_scale"])
        # self.log_metrics(
        #     outputs, trainer_instance.global_val_step, trainer_instance, "val"
        # )

    def on_validation_epoch_end(self, trainer_instance=None):

        forces_gt = torch.cat(self.val_gt, dim=0).cpu().numpy()
        forces_pred = torch.cat(self.val_pred, dim=0).cpu().numpy()
        force_scale = torch.cat(self.val_force_scale, dim=0).cpu().numpy()

        forces_gt = forces_gt * force_scale
        forces_pred = forces_pred * force_scale

        rmse = np.sqrt(
            np.mean((forces_gt - forces_pred) ** 2, axis=0)
        ) * 1000  # Convert to mN
        loss = rmse.sum()

        outputs = {
            "loss": loss,
            "rmse_xyz": rmse
        }

        if loss < self.min_force_loss:
            self.min_force_loss = loss
            self.min_force_rmse = rmse

        print(f"Validation RMSE: Fx: {rmse[0]:.2f} mN, Fy: {rmse[1]:.2f} mN, Fz: {rmse[2]:.2f} mN, Total: {loss:.2f} mN")
        print(f"Minimum RMSE: Fx: {self.min_force_rmse[0]:.2f} mN, Fy: {self.min_force_rmse[1]:.2f} mN, Fz: {self.min_force_rmse[2]:.2f} mN, Total: {self.min_force_loss:.2f} mN")

        self.log_metrics(outputs, trainer_instance.global_val_step, trainer_instance, "val")

        im_corr = plot_correlation(forces_gt, forces_pred)
        img_err, img_cone = plot_forces_error(forces_gt, forces_pred)

        if trainer_instance is not None:
            trainer_instance.wandb.log(
                {
                    "val/correlation": trainer_instance.wandb.Image(im_corr),
                    "val/error": trainer_instance.wandb.Image(img_err),
                    "val/error_cone": trainer_instance.wandb.Image(img_cone),
                }
            )

        self.val_pred = []
        self.val_gt = []
        self.val_force_scale = []
