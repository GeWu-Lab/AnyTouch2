# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import matplotlib.pyplot as plt

from typing import Optional
from tactile_ssl import algorithm

import torch
import torch.nn.functional as F
from tqdm import tqdm
from scipy import stats

from .test_task import TestTaskSL
from sparsh.tactile_ssl.utils.plotting_forces import (
    plot_correlation,
    plot_forces_error,
)
from PIL import Image


class TestForceSL(TestTaskSL):
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
        forces_gt = []
        forces_pred = []
        path_save_outputs = f"{self.path_outputs}/{self.epoch}_predictions.npy"
        # check if the outputs are already saved
        if os.path.exists(path_save_outputs):
            return

        iter = 0
        for sample in tqdm(dataloader):
            x = sample["image"].to(self.device)
            y_gt = sample["force"].to(self.device)
            y_pred = self.module(x)
            forces_gt.append(y_gt.cpu().detach().numpy())
            forces_pred.append(y_pred.cpu().detach().numpy())

            del x, y_gt, y_pred
            torch.cuda.empty_cache()

        forces_gt = np.vstack(forces_gt)
        forces_pred = np.vstack(forces_pred)

        outputs = {
            "forces_gt": forces_gt,
            "forces_pred": forces_pred,
        }

        np.save(f"{self.path_outputs}/{self.epoch}_predictions.npy", outputs)

    def get_overall_metrics(self, dataset, over_all_outputs=False):
        scale = dataset.max_abs_forceXYZ
        if not over_all_outputs:
            outputs = np.load(
                f"{self.path_outputs}/{self.epoch}_predictions.npy", allow_pickle=True
            ).item()
        else:
            # load all outputs that start with self.task
            outputs = {}
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
            for key in outputs.keys():
                outputs[key] = np.vstack(outputs[key])

        forces_gt = outputs["forces_gt"] * scale  # in N
        forces_pred = outputs["forces_pred"] * scale  # in N

        rmse = np.sqrt((forces_gt - forces_pred) ** 2).mean(axis=1).mean()
        rmse_std = np.sqrt((forces_gt - forces_pred) ** 2).std(axis=1).mean()
        corr = np.array(
            [stats.pearsonr(forces_gt[:, i], forces_pred[:, i])[0] for i in range(3)]
        )
        sem = rmse_std / np.sqrt(forces_gt.shape[0])
        ci95 = 1.96 * rmse_std / np.sqrt(forces_gt.shape[0])

        if not over_all_outputs:
            print("Metrics for {self.task}_{self.dataset_name}:")
        else:
            print("Metrics for all outputs:")

        print(f"RMSE: {rmse} ± {rmse_std} N")
        print(f"Correlation: {corr}")
        print(f"Total samples: {forces_gt.shape[0]}")

        metrics = {
            "rmse": rmse,
            "rmse_std": rmse_std,
            "corr": corr,
            "sem": sem,
            "ci95": ci95,
            "n_samples": forces_gt.shape[0],
        }
        if over_all_outputs:
            np.save(f"{self.path_output_model}/{self.epoch}_metrics.npy", metrics)
        else:
            np.save(
                f"{self.path_output_model}/{self.epoch}_{self.dataset_name}_metrics.npy",
                metrics,
            )

    def make_plots(self, dataset, params: Optional[dict] = None):
        outputs = np.load(
            f"{self.path_outputs}/{self.epoch}_predictions.npy", allow_pickle=True
        ).item()

        scale = dataset.max_abs_forceXYZ
        forces_gt = outputs["forces_gt"] * scale  # in N
        forces_pred = outputs["forces_pred"] * scale  # in N

        img_corr = plot_correlation(forces_gt, forces_pred)
        img_err, img_cone = plot_forces_error(forces_gt, forces_pred)

        img_corr.save(f"{self.path_outputs}/{self.epoch}_correlation.png")
        img_err.save(f"{self.path_outputs}/{self.epoch}_XYZerror.png")
        img_cone.save(f"{self.path_outputs}/{self.epoch}_ConeError.png")