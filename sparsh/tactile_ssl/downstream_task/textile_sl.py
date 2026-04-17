# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Tuple, Optional, List, Union
from functools import partial
import matplotlib.pyplot as plt
from PIL import Image
import io

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sparsh.tactile_ssl.utils.logging import get_pylogger
from sparsh.tactile_ssl.downstream_task.sl_module import SLModule
from sparsh.tactile_ssl.downstream_task.attentive_pooler import AttentivePooler
from sparsh.tactile_ssl.model import VIT_EMBED_DIMS

from sklearn.metrics import (
    top_k_accuracy_score,
    accuracy_score,
    balanced_accuracy_score,
)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

log = get_pylogger(__name__)


class TextileLinearProbe(nn.Module):
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
        num_classes=20,
    ):
        super().__init__()

        self.num_classes = num_classes
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
            nn.Linear(embed_dim // 4, num_classes),
        )

    def forward(self, x):
        x = self.pooler(x).squeeze(1)
        x = self.probe(x)
        return x


class TextileSLModule(SLModule):
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
        weights_classes: Optional[list] = None,
        class_labels: Optional[list] = None,
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
        self.val_outputs = []
        self.val_batches = []
        self.n_classes = model_task.num_classes
        self.weights_classes = torch.tensor([1.0]*self.n_classes) if weights_classes is None else torch.tensor(weights_classes)
        self.class_labels = class_labels
        assert self.n_classes == len(self.class_labels) == len(self.weights_classes)

    def forward(self, x: torch.Tensor):
        z = self.model_encoder(x)
        if self.train_encoder:
            y_pred = self.model_task(z)
        else:
            y_pred = self.model_task(z.detach())
        return y_pred

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        x = batch["image"]
        y_gt = batch["textile_label"]
        self.weights_classes = self.weights_classes.to(y_gt.device)

        y_pred = self.forward(x)
        loss = F.cross_entropy(y_pred, y_gt, weight=self.weights_classes)
        y_pred = F.softmax(y_pred, dim=1)
        y_pred = y_pred.detach()

        y_gt_np = y_gt.cpu().numpy()
        y_pred_np = y_pred.cpu().numpy()
        y_pred_labels = y_pred_np.argmax(axis=1)
        pred_labels_chance = np.random.choice(self.n_classes, size=y_gt_np.shape[0])
        accuracy = accuracy_score(y_gt_np, y_pred_labels)
        accuracy_chance = accuracy_score(y_gt_np, pred_labels_chance)
        top_k_accuracy = top_k_accuracy_score(
                y_gt_np, y_pred_np, k=3, labels=range(self.n_classes)
            )
        balanced_accuracy = balanced_accuracy_score(y_gt_np, y_pred_labels)
        return {
            "loss": loss,
            "accuracy": accuracy,
            "accuracy_chance": accuracy_chance,
            "top_k_accuracy": top_k_accuracy,
            "balanced_accuracy": balanced_accuracy,
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
            for label_metric in ["accuracy", "accuracy_chance", "top_k_accuracy", "balanced_accuracy"]:
                trainer_instance.wandb.log(
                    {
                        f"{label}/{label_metric}": outputs[label_metric],
                        f"global_{label}_step": step,
                    }
                )

    def on_train_batch_end(self, outputs, batch, batch_idx, trainer_instance=None):
        self.log_metrics(outputs, trainer_instance.global_step, trainer_instance)

    def on_validation_batch_end(
        self, outputs: Dict, batch: Dict, batch_idx: int, trainer_instance=None
    ):
        self.val_outputs.append(outputs)
        self.val_batches.append(batch)
        self.log_metrics(
            outputs, trainer_instance.global_val_step, trainer_instance, "val"
        )

    def on_validation_epoch_end(self, trainer_instance=None):
        textile_gt = (
            torch.cat([batch["textile_label"] for batch in self.val_batches], dim=0)
            .cpu()
            .numpy()
        )
        textile_pred = (
            torch.cat([output["y_pred"] for output in self.val_outputs], dim=0)
            .cpu()
            .numpy()
        )
        textile_pred = textile_pred.argmax(axis=1)

        cm = confusion_matrix(textile_gt, textile_pred, normalize="true", labels=range(self.n_classes))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=self.class_labels
        )

        disp.plot(xticks_rotation="vertical", cmap="Blues")
        fig = disp.ax_.get_figure()
        fig.set_figwidth(10)
        fig.set_figheight(10)
        disp.im_.set_clim(0, 1)
        plt.tight_layout()
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

        self.val_outputs = []
        self.val_batches = []