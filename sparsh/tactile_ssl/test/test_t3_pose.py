# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Optional
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from sklearn.metrics import (
    top_k_accuracy_score,
    accuracy_score,
    balanced_accuracy_score,
)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tactile_ssl import algorithm
from .test_task import TestTaskSL

class TestPoseSL(TestTaskSL):
    def __init__(
        self,
        device,
        module: algorithm.Module,
    ):
        super().__init__(
            device=device,
            module=module,
        )

    def run_model(self, dataset, dataloader):
        y_gt = {}
        y_pred = {}

        path_save_outputs = f"{self.path_outputs}/{self.epoch}_predictions.npy"
        if os.path.exists(path_save_outputs):
            return

        for sample in tqdm(dataloader):
            x = sample["image"]
            gt = sample["pose_labels"]

            if len(y_gt) == 0:
                y_gt = {key: [] for key in gt.keys()}
                y_pred = {key: [] for key in gt.keys()}

            for key in x.keys():
                x[key] = x[key].to(self.device)

            pred = self.module(x)

            for key in gt.keys():
                y_gt[key].append(gt[key])
                y_pred[key].append(pred[key])

        y_gt = {key: torch.cat(y_gt[key], dim=0) for key in y_gt.keys()}
        y_pred = {key: torch.cat(y_pred[key], dim=0) for key in y_pred.keys()}
        y_pred_probs = {
            key: F.softmax(y_pred[key], dim=1).cpu() for key in y_pred.keys()
        }
        pose_pred_labels = {key: y_pred[key].argmax(dim=1) for key in y_pred.keys()}

        # get metrics
        y_pred = {key: y_pred[key].cpu().numpy() for key in y_pred.keys()}
        y_gt = {key: y_gt[key].cpu().numpy() for key in y_gt.keys()}
        pose_pred_labels = {
            key: pose_pred_labels[key].cpu().numpy() for key in pose_pred_labels.keys()
        }

        outputs = {
            "y_gt": y_gt,
            "y_pred": y_pred_probs,
            "pose_pred_labels": pose_pred_labels,
        }
        np.save(f"{self.path_outputs}/{self.epoch}_predictions.npy", outputs)
    
    def get_accuracy(self, data):
        acc_tx = data["y_gt"]["tx"] == data["pose_pred_labels"]["tx"]
        acc_ty = data["y_gt"]["ty"] == data["pose_pred_labels"]["ty"]
        acc_yaw = data["y_gt"]["yaw"] == data["pose_pred_labels"]["yaw"]
        acc = np.mean([acc_tx, acc_ty, acc_yaw])
        return acc

    def get_overall_metrics(self, dataset, over_all_outputs=False):
        n_classes = self.module.model_task.num_classes

        if not over_all_outputs:
            outputs = np.load(
                f"{self.path_outputs}/{self.epoch}_predictions.npy", allow_pickle=True
            ).item()
        else:
            # load all outputs that start with self.task
            outputs = {}
            test_datasets_acc = []
            for batch in os.listdir(self.path_output_model):
                if os.path.isdir(f"{self.path_output_model}/{batch}"):
                    output = np.load(
                        f"{self.path_output_model}/{batch}/{self.epoch}_predictions.npy",
                        allow_pickle=True,
                    ).item()
                    for key in output.keys():
                        if key not in outputs:
                            outputs[key] = {}
                            for ax in output[key].keys():
                                outputs[key][ax] = []
                        for ax in output[key].keys():
                            outputs[key][ax].append(output[key][ax])
                    
                    test_datasets_acc.append(self.get_accuracy(output))

            for key in outputs.keys():
                for ax in output[key].keys():
                    outputs[key][ax] = np.concatenate(outputs[key][ax])

        labels_gt = outputs["y_gt"]
        labels_pred = outputs["pose_pred_labels"]
        probs_pred = outputs["y_pred"]

        labels_chance = {
            key: np.random.randint(0, n_classes, labels_gt[key].shape)
            for key in labels_gt.keys()
        }
        probs_chance = {
            key: np.random.rand(*probs_pred[key].shape) for key in probs_pred.keys()
        }

        metrics = {}
        for key in labels_gt.keys():
            metrics[key] = {}
            metrics[key]["top_k_accuracy"] = top_k_accuracy_score(
                labels_gt[key], probs_pred[key], k=3, labels=range(n_classes)
            )
            metrics[key]["accuracy"] = accuracy_score(labels_gt[key], labels_pred[key])
            metrics[key]["balanced_accuracy"] = balanced_accuracy_score(
                labels_gt[key],
                labels_pred[key],
            )
            metrics[key]["top_k_accuracy_chance"] = top_k_accuracy_score(
                labels_gt[key], probs_chance[key], k=3, labels=range(n_classes)
            )
            metrics[key]["accuracy_chance"] = accuracy_score(
                labels_gt[key], labels_chance[key]
            )
            metrics[key]["balanced_accuracy_chance"] = balanced_accuracy_score(
                labels_gt[key],
                labels_chance[key],
            )

        metrics["avg_top_k_accuracy"] = np.mean(
            [metrics[key]["top_k_accuracy"] for key in labels_gt.keys()]
        )
        metrics["avg_accuracy"] = np.mean(
            [metrics[key]["accuracy"] for key in labels_gt.keys()]
        )
        metrics["avg_balanced_accuracy"] = np.mean(
            [metrics[key]["balanced_accuracy"] for key in labels_gt.keys()]
        )

        metrics["avg_top_k_accuracy_chance"] = np.mean(
            [metrics[key]["top_k_accuracy_chance"] for key in labels_gt.keys()]
        )
        metrics["avg_accuracy_chance"] = np.mean(
            [metrics[key]["accuracy_chance"] for key in labels_gt.keys()]
        )
        metrics["avg_balanced_accuracy_chance"] = np.mean(
            [metrics[key]["balanced_accuracy_chance"] for key in labels_gt.keys()]
        )

        metrics["n_samples"] = labels_gt["tx"].shape[0]

        # compute 95% confidence interval
        test_datasets_avg_acc = np.mean(test_datasets_acc)
        test_datasets_sd_acc = np.std(test_datasets_acc)
        test_datasets_ci_acc = 1.96 * test_datasets_sd_acc / np.sqrt(metrics["n_samples"])

        metrics['test_datasets_avg_acc'] = test_datasets_avg_acc
        metrics['test_datasets_sd_acc'] = test_datasets_sd_acc
        metrics['test_datasets_ci95'] = test_datasets_ci_acc

        self.plot_confusion_matrix(labels_gt, labels_pred, n_classes)

        np.save(f"{self.path_output_model}/{self.epoch}_metrics.npy", metrics)

    def plot_confusion_matrix(self, labels_gt, labels_pred, n_classes):
        labels_xy = self.module.labels_xy
        labels_py = self.module.labels_py

        fig, ax = plt.subplots(1, 3, figsize=(20, 7))
        fig.subplots_adjust(
            left=0.078, right=0.989, top=0.969, bottom=0.199, hspace=0.235, wspace=0.384
        )
        # fig.tight_layout()
        for i, key in enumerate(labels_gt.keys()):
            labels_txt = labels_xy if key in ["tx", "ty"] else labels_py
            cm = confusion_matrix(
                labels_gt[key],
                labels_pred[key],
                normalize="true",
                labels=range(n_classes),
            )
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm, display_labels=labels_txt
            )
            disp.plot(
                ax=ax[i], xticks_rotation="vertical", cmap="Blues", values_format=".2f",
            )
            disp.im_.set_clim(0, 1)
            disp.im_.colorbar.remove()
            ax[i].set_title(r"$\Delta$" + f"_{key}")

        plt.savefig(f"{self.path_output_model}/{self.epoch}_confusion_matrix.png")