# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Tuple, Optional, List, Union
from matplotlib.axes import SubplotBase
from numpy.typing import NDArray
from functools import partial

import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from PIL import Image
import io

import torch
import torch.nn as nn
import torch.nn.functional as F

from sparsh.tactile_ssl.utils.logging import get_pylogger
from sparsh.tactile_ssl.downstream_task.sl_module import SLModule
from sparsh.tactile_ssl.downstream_task.attentive_pooler import AttentivePooler
from sparsh.tactile_ssl.utils.plotting_forces import plot_correlation, plot_forces_error

log = get_pylogger(__name__)


class SlipSLModule(SLModule):
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
        input_delta_force: Optional[bool] = False,
        predict_delta_force: Optional[bool] = False,
        weights_classes: Optional[NDArray] = [1.0, 1.0],
        add_batch_norm_probe: Optional[bool] = False,
        is_anytouch=False,
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
        )
        self.val_gt_labels = []
        self.val_pred_labels = []
        self.val_gt_forces = []
        self.val_pred_forces = []
        self.val_delta_force_scale = []
        self.min_force_rmse = [float("inf"), float("inf"), float("inf")]
        self.min_force_loss = float("inf")
        self.max_acc = 0.0
        self.max_f1 = 0.0

        assert not (input_delta_force and predict_delta_force), ValueError(
            "Cannot have both input_delta_force and predict_delta_force"
        )

        self.input_delta_force = input_delta_force
        self.predict_delta_force = predict_delta_force
        self.weights_classes = torch.tensor(weights_classes)
        if add_batch_norm_probe:
            self.model_task.probe = nn.Sequential(
                nn.BatchNorm1d(model_task.probe.in_features, affine=False, eps=1e-6),
                self.model_task.probe,
            )

    def forward(self, imgs: torch.Tensor, force: torch.Tensor = None):
        z = self.model_encoder(imgs)
        inputs = {"latent": z}
        if self.input_delta_force:
            inputs["force"] = force
        output = self.model_task(inputs)
        return output

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        imgs = batch["image"]
        slip_gt = batch["slip_label"]
        force_gt = batch["delta_force"]
        self.weights_classes = self.weights_classes.to(imgs.device)
        slip_baseline = torch.zeros_like(slip_gt).to(slip_gt.device)

        out_pred = self.forward(imgs, force_gt)
        output = {}
        loss = None

        # compute loss for slip
        if "slip" in out_pred:
            slip_pred = out_pred["slip"]
            loss = F.cross_entropy(slip_pred, slip_gt, weight=self.weights_classes)
            accuracy = (slip_pred.argmax(1) == slip_gt).float().mean()
            accuracy_baseline = (slip_baseline == slip_gt).float().mean()
            f1 = f1_score(
                slip_gt.cpu().numpy(),
                slip_pred.argmax(1).cpu().numpy(),
                zero_division=0,
            )

            output = {
                "slip_accuracy": accuracy,
                "slip_accuracy_baseline": accuracy_baseline,
                "slip_f1_score": f1,
                "slip_pred": slip_pred.detach(),
            }

        if self.predict_delta_force:
            force_pred = out_pred["force"]
            loss_force = F.smooth_l1_loss(force_pred, force_gt)
            rmse_xyz = torch.sqrt(
                F.mse_loss(
                    force_pred.detach(), force_gt.detach(), reduction="none"
                ).mean(dim=0)
            )
            loss += loss_force
            output["rmse_∆Fx"] = torch.sqrt(rmse_xyz[0])
            output["rmse_∆Fy"] = torch.sqrt(rmse_xyz[1])
            output["rmse_∆Fz"] = torch.sqrt(rmse_xyz[2])
            output["force_pred"] = force_pred.detach()

        output["loss"] = loss
        return output

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        return self.training_step(batch, batch_idx)

    def log_metrics(self, outputs, step, trainer_instance=None, label="train"):
        if trainer_instance is not None:
            for key, value in outputs.items():
                trainer_instance.wandb.log(
                    {
                        f"{label}/{key}": value,
                        f"global_{label}_step": step,
                    }
                )

    def on_train_batch_end(self, outputs, batch, batch_idx, trainer_instance=None):
        self.log_metrics(outputs, trainer_instance.global_step, trainer_instance)

    def on_validation_batch_end(
        self, outputs: Dict, batch: Dict, batch_idx: int, trainer_instance=None
    ):
        if "slip_pred" in outputs:
            self.val_gt_labels.append(batch["slip_label"])
            self.val_pred_labels.append(outputs["slip_pred"].argmax(1))

        if self.predict_delta_force:
            self.val_gt_forces.append(batch["delta_force"])
            self.val_pred_forces.append(outputs["force_pred"])
            self.val_delta_force_scale.append(batch["delta_force_scale"])

        self.log_metrics(
            outputs, trainer_instance.global_val_step, trainer_instance, "val"
        )

    def on_validation_epoch_end(self, trainer_instance=None):
        if len(self.val_pred_labels) > 0:
            self.show_val_slip(trainer_instance)
        if self.predict_delta_force:
            self.show_val_forces(trainer_instance)

    def show_val_slip(self, trainer_instance=None):
        grasp_gt = torch.cat(self.val_gt_labels, dim=0).cpu().numpy()
        grasp_pred = torch.cat(self.val_pred_labels, dim=0).cpu().numpy()

        cm = confusion_matrix(grasp_gt, grasp_pred, normalize="true", labels=[0, 1])
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=["no_slip", "slip"]
        )

        pred = torch.tensor(grasp_pred)
        gt = torch.tensor(grasp_gt)
        acc = (pred == gt).float().mean()
        f1 = f1_score(
                gt.cpu().numpy(),
                pred.cpu().numpy(),
                zero_division=0,
            )
        
        if acc > self.max_acc:
            self.max_acc = acc
        if f1 > self.max_f1:
            self.max_f1 = f1

        print(f"Validation Accuracy: {acc:.5f}, F1 Score: {f1:.5f}")
        print(f"Maximum Accuracy: {self.max_acc:.5f}, Maximum F1 Score: {self.max_f1:.5f}")

        if trainer_instance is not None:
            trainer_instance.wandb.log(
                {
                    "test/accuracy": acc,
                    "test/f1_score": f1,
                    "global_test_step": trainer_instance.global_val_step,
                }
            )

        disp.plot()
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format="png")
        plt.close("all")
        im = Image.open(img_buf)

        if trainer_instance is not None:
            trainer_instance.wandb.log(
                {
                    "val/cm": trainer_instance.wandb.Image(im),
                }
            )
        plt.close()

        self.val_pred_labels = []
        self.val_gt_labels = []

    def show_val_forces(self, trainer_instance=None):
        forces_gt = torch.cat(self.val_gt_forces, dim=0).cpu().numpy()
        forces_pred = torch.cat(self.val_pred_forces, dim=0).cpu().numpy()
        force_scale = torch.cat(self.val_delta_force_scale, dim=0).cpu().numpy()

        forces_gt = forces_gt * force_scale
        forces_pred = forces_pred * force_scale

        rmse = np.sqrt(
            np.mean((forces_gt - forces_pred) ** 2, axis=0)
        ) * 1000  # Convert to mN
        loss = rmse.sum()

        if loss < self.min_force_loss:
            self.min_force_loss = loss
            self.min_force_rmse = rmse
        
        print(f"Validation RMSE: Fx: {rmse[0]:.2f} mN, Fy: {rmse[1]:.2f} mN, Fz: {rmse[2]:.2f} mN, Total: {loss:.2f} mN")
        print(f"Minimum RMSE: Fx: {self.min_force_rmse[0]:.2f} mN, Fy: {self.min_force_rmse[1]:.2f} mN, Fz: {self.min_force_rmse[2]:.2f} mN, Total: {self.min_force_loss:.2f} mN")


        if trainer_instance is not None:
            trainer_instance.wandb.log(
                {
                    "test/rmse_∆Fx": rmse[0],
                    "test/rmse_∆Fy": rmse[1],
                    "test/rmse_∆Fz": rmse[2],
                    "test/rmse_xyz": loss,
                    "global_test_step": trainer_instance.global_val_step,
                }
            )

        im_corr = plot_correlation(forces_gt, forces_pred)
        im_err, _ = plot_forces_error(forces_gt, forces_pred)

        if trainer_instance is not None:
            trainer_instance.wandb.log(
                {
                    "val/correlation": trainer_instance.wandb.Image(im_corr),
                    "val/error": trainer_instance.wandb.Image(im_err),
                }
            )

        self.val_gt_forces = []
        self.val_pred_forces = []
        self.val_delta_force_scale = []
