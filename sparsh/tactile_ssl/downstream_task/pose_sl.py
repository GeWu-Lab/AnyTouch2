# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Tuple, Optional, List, Union
from matplotlib.axes import SubplotBase
from numpy.typing import NDArray
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


class PoseLinearProbe(nn.Module):
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
        num_classes=10,
        num_input_fingers=1,
        is_anytouch=False,
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

        embed_dim = embed_dim * num_input_fingers
        self.probe_tx = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, num_classes),
        )
        self.probe_ty = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, num_classes),
        )
        self.probe_yaw = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, num_classes),
        )

    def forward(self, x):
        x_pool = []
        for i in range(len(x)):
            x_pool.append(self.pooler(x[i]).squeeze(1))

        x_pool = torch.cat(x_pool, dim=1)
        y_tx = self.probe_tx(x_pool)
        y_ty = self.probe_ty(x_pool)
        y_yaw = self.probe_yaw(x_pool)

        output = {
            "tx": y_tx,
            "ty": y_ty,
            "yaw": y_yaw,
        }
        return output


class PoseSLModule(SLModule):
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
        weights_classes_tx: Optional[list] = None,
        weights_classes_ty: Optional[list] = None,
        weights_classes_yaw: Optional[list] = None,
        bins_translation: Optional[list] = None,
        bins_rotation: Optional[list] = None,
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
        self.max_acc = 0
        self.max_bal_acc = 0
        self.val_pred = {}
        self.val_gt = {}
        self.n_classes = model_task.num_classes
        self.weights_classes = {
            "tx": (
                torch.tensor(weights_classes_tx)
                if weights_classes_tx is not None
                else None
            ),
            "ty": (
                torch.tensor(weights_classes_ty)
                if weights_classes_ty is not None
                else None
            ),
            "yaw": (
                torch.tensor(weights_classes_yaw)
                if weights_classes_yaw is not None
                else None
            ),
        }

        ths_xy = np.array(bins_translation)
        ths_py = np.array(bins_rotation)
        ths_py = np.concatenate([ths_py[::-1] * -1, ths_py])
        ths_xy = np.concatenate([ths_xy[::-1] * -1, ths_xy])

        # create the labels for the rotation and translation
        self.labels_xy = []
        for i in range(len(ths_xy)):
            if i == 0:
                self.labels_xy.append(r"$<$" + f"{ths_xy[i]}")
            else:
                self.labels_xy.append(f"[{ths_xy[i - 1]}, {ths_xy[i]})")
        self.labels_xy.append(r"$>=$" + f"{ths_xy[-1]}")

        self.labels_py = []
        for i in range(len(ths_py)):
            if i == 0:
                self.labels_py.append(r"$<$" + f"{ths_py[i]}")
            else:
                self.labels_py.append(f"[{ths_py[i - 1]}, {ths_py[i]})")
        self.labels_py.append(r"$>=$" + f"{ths_py[-1]}")

    def forward(self, inputs: dict):
        x = []
        for key in inputs.keys():
            x.append(self.model_encoder(inputs[key]))

        if self.train_encoder:
            y_pred = self.model_task(x)
        else:
            x = [z.detach() for z in x]
            y_pred = self.model_task(x)
        return y_pred

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        x = batch["image"]
        y_gt = batch["pose_labels"]

        y_pred = self.forward(x)
        losses = {}
        metrics = {}
        for key in y_gt.keys():
            losses[key] = F.cross_entropy(
                y_pred[key],
                y_gt[key],
                weight=self.weights_classes[key].to(y_gt[key].device),
            )
        loss = sum(losses.values())

        y_pred = {key: y_pred[key].detach() for key in y_pred.keys()}
        pose_pred_labels = {key: y_pred[key].argmax(dim=1) for key in y_pred.keys()}
        outputs = {
            "loss": loss,
            "losses": losses,
            "pose_pred_labels": pose_pred_labels,
            "metrics": None,
        }

        # send to cpu for metrics computation
        y_pred = {key: y_pred[key].cpu().numpy() for key in y_pred.keys()}
        y_gt = {key: y_gt[key].cpu().numpy() for key in y_gt.keys()}
        pose_pred_labels = {
            key: pose_pred_labels[key].cpu().numpy() for key in pose_pred_labels.keys()
        }
        metrics = {}
        for key in y_gt.keys():
            metrics[key] = {}
            metrics[key]["top_k_accuracy"] = top_k_accuracy_score(
                y_gt[key], y_pred[key], k=3, labels=range(self.n_classes)
            )
            metrics[key]["accuracy"] = accuracy_score(y_gt[key], pose_pred_labels[key])
            metrics[key]["balanced_accuracy"] = balanced_accuracy_score(
                y_gt[key],
                pose_pred_labels[key],
            )

        outputs["metrics"] = metrics

        return outputs

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        return self.training_step(batch, batch_idx)

    def log_metrics(self, outputs, step, trainer_instance=None, label="train"):
        if trainer_instance is not None:
            for key in outputs["losses"].keys():
                trainer_instance.wandb.log(
                    {
                        f"{label}/{key}/loss": outputs["losses"][key],
                        f"global_{label}_step": step,
                    }
                )
            for key in outputs["metrics"].keys():
                for metric in outputs["metrics"][key].keys():
                    trainer_instance.wandb.log(
                        {
                            f"{label}/{key}/{metric}": outputs["metrics"][key][metric],
                            f"global_{label}_step": step,
                        }
                    )

    def on_train_batch_end(self, outputs, batch, batch_idx, trainer_instance=None):
        self.log_metrics(outputs, trainer_instance.global_step, trainer_instance)

    def on_validation_batch_end(
        self, outputs: Dict, batch: Dict, batch_idx: int, trainer_instance=None
    ):
        if len(self.val_pred) == 0:
            for key in batch["pose_labels"].keys():
                self.val_pred[key] = []
                self.val_gt[key] = []

        for key in batch["pose_labels"].keys():
            self.val_pred[key].append(outputs["pose_pred_labels"][key])
            self.val_gt[key].append(batch["pose_labels"][key])

        self.log_metrics(
            outputs, trainer_instance.global_val_step, trainer_instance, "val"
        )

    def on_validation_epoch_end(self, trainer_instance=None):
        labels_gt = []
        labels_pred = []
        all_acc = []
        all_bal_acc = []
        all_top_k_acc = []
        for key in self.val_gt.keys():
            labels_gt = torch.cat(self.val_gt[key], dim=0).cpu().numpy()
            labels_pred = torch.cat(self.val_pred[key], dim=0).cpu().numpy()

            labels_txt = self.labels_xy if key in ["tx", "ty"] else self.labels_py

            # compute confusion matrix
            cm = confusion_matrix(
                labels_gt, labels_pred, normalize="true", labels=range(self.n_classes)
            )
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=labels_txt,
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
                        f"val/{key}/cm": trainer_instance.wandb.Image(im),
                    }
                )

                acc = accuracy_score(labels_gt, labels_pred)
                bal_acc = balanced_accuracy_score(
                    labels_gt,
                    labels_pred,
                )

                all_acc.append(acc)
                all_bal_acc.append(bal_acc)

                trainer_instance.wandb.log(
                    {
                        f"test/{key}/accuracy": acc,
                        f"test/{key}/balanced_accuracy": bal_acc,
                        f"global_test_step": trainer_instance.global_val_step,
                    }
                )

        if trainer_instance is not None:
            now_acc = np.mean(all_acc)
            now_bal_acc = np.mean(all_bal_acc)
            trainer_instance.wandb.log(
                {
                    "test/overall/accuracy": now_acc,
                    "test/overall/balanced_accuracy": now_bal_acc,
                    "global_test_step": trainer_instance.global_val_step,
                }
            )
            if now_acc > self.max_acc:
                self.max_acc = now_acc
            if now_bal_acc > self.max_bal_acc:
                self.max_bal_acc = now_bal_acc
            
            print(f"Now Accuracy: {now_acc:.5f}, Now Balanced Accuracy: {now_bal_acc:.5f}")
            print(f"Max Accuracy: {self.max_acc:.5f}, Max Balanced Accuracy: {self.max_bal_acc:.5f}")


        self.val_pred = {}
        self.val_gt = {}