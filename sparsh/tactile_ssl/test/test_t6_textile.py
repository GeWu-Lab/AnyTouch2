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
    classification_report
)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tactile_ssl import algorithm
from .test_task import TestTaskSL

class TestTextileSL(TestTaskSL):
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
        y_gt = []
        y_pred = []

        path_save_outputs = f"{self.path_outputs}/{self.epoch}_predictions.npy"
        if os.path.exists(path_save_outputs):
            return

        for sample in tqdm(dataloader):
            x = sample["image"].to(self.device)
            gt = sample["textile_label"]

            pred = self.module(x)
            y_gt.append(gt)
            y_pred.append(pred)
        
        y_gt = torch.cat(y_gt, dim=0).cpu().numpy()
        y_pred = torch.cat(y_pred, dim=0)
        y_pred_probs = F.softmax(y_pred, dim=1).cpu().numpy()
        y_pred_label =  y_pred.argmax(dim=1).cpu().numpy()

        outputs = {
            "y_gt": y_gt,
            "y_pred": y_pred_probs,
            "y_pred_label": y_pred_label,
        }
        np.save(f"{self.path_outputs}/{self.epoch}_predictions.npy", outputs)
    
    def get_accuracy(self, data):
        y_gt = data["y_gt"]
        y_pred_label = data["y_pred_label"]
        acc = np.mean(y_gt==y_pred_label)
        return acc

    def get_overall_metrics(self, dataset, over_all_outputs=False):
        n_classes = self.module.n_classes

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
                            outputs[key] = []
                        outputs[key].append(output[key])
                    
                    test_datasets_acc.append(self.get_accuracy(output))
            
            for key in outputs.keys():
                outputs[key] = np.concatenate(outputs[key], axis=0)

        labels_gt = outputs["y_gt"]
        labels_pred = outputs["y_pred_label"]
        probs_pred = outputs["y_pred"]

        labels_chance = np.random.randint(0, n_classes, labels_gt.shape)
        # probs_chance = np.random.rand(*probs_pred.shape)

        metrics = {}
        metrics["accuracy"] = accuracy_score(labels_gt, labels_pred)
        metrics["accuracy_chance"] = accuracy_score(labels_gt, labels_chance)
        metrics["top_k_accuracy"] = top_k_accuracy_score(labels_gt, probs_pred, k=3, labels=range(n_classes))
        metrics["balanced_accuracy"] = balanced_accuracy_score(labels_gt, labels_pred)
        metrics["n_samples"] = labels_gt.shape[0]

        # compute 95% confidence interval
        test_datasets_avg_acc = np.mean(test_datasets_acc)
        test_datasets_sd_acc = np.std(test_datasets_acc)
        test_datasets_ci_acc = 1.96 * test_datasets_sd_acc / np.sqrt(metrics["n_samples"])

        metrics['test_datasets_avg_acc'] = test_datasets_avg_acc
        metrics['test_datasets_sd_acc'] = test_datasets_sd_acc
        metrics['test_datasets_ci95'] = test_datasets_ci_acc
        # metrics['report'] = classification_report(labels_gt, labels_pred, target_names=["no_stable", "stable"], output_dict=True)
        
        self.plot_confusion_matrix(labels_gt, labels_pred)
        np.save(f"{self.path_output_model}/{self.epoch}_metrics.npy", metrics)
    
    def plot_confusion_matrix(self, labels_gt, labels_pred):
        labels = self.module.class_labels
        fig, ax = plt.subplots(figsize=(10, 10))
        cm = confusion_matrix(
                labels_gt,
                labels_pred,
                normalize="true",
                labels=range(len(labels)),
            )
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(ax=ax, xticks_rotation="vertical", cmap="Blues", values_format=".2f")
        fig = disp.ax_.get_figure()
        disp.im_.set_clim(0, 1)
        plt.tight_layout()
        plt.savefig(f"{self.path_output_model}/{self.epoch}_confusion_matrix.png")
        plt.close()