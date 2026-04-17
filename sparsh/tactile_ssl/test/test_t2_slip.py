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
from scipy import stats
from scipy.spatial.transform import Rotation as R

from sklearn.metrics import (
    classification_report,
    balanced_accuracy_score,
)
from collections import deque
from tactile_ssl import algorithm
from .test_task import TestTaskSL

sns.set_theme(context="paper", style="dark", font_scale=1.0)

class TestSlipSL(TestTaskSL):
    def __init__(
        self,
        device,
        module: algorithm.Module,
    ):
        super().__init__(
            device=device,
            module=module,
        )
        self.offset = 5
        self.th = 0.4

    def run_model(self, dataset, dataloader):

        delta_forces_gt = []
        delta_forces_pred = []
        slip_labels_gt = []
        slip_prob_pred = []

        path_save_outputs = f"{self.path_outputs}/{self.epoch}_predictions.npy"
        if os.path.exists(path_save_outputs):
            return

        for sample in tqdm(dataloader):
            x = sample["image"].to(self.device)
            slip_gt = sample["slip_label"]
            force_gt = sample["delta_force"]

            out_pred = self.module(x)
            slip_pred = out_pred["slip"]
            slip_pred = F.softmax(slip_pred, dim=1)
            delta_force_pred = out_pred["force"]

            delta_forces_gt.append(force_gt.cpu().detach().numpy())
            delta_forces_pred.append(delta_force_pred.cpu().detach().numpy())
            slip_labels_gt.append(slip_gt.cpu().detach().numpy())
            slip_prob_pred.append(slip_pred.cpu().detach().numpy())

        delta_forces_gt = np.concatenate(delta_forces_gt)
        delta_forces_pred = np.concatenate(delta_forces_pred)
        slip_labels_gt = np.concatenate(slip_labels_gt)
        slip_prob_pred = np.concatenate(slip_prob_pred)

        outputs = {
            "delta_forces_gt": delta_forces_gt,
            "delta_forces_pred": delta_forces_pred,
            "slip_label_gt": slip_labels_gt,
            "slip_prob_pred": slip_prob_pred,
        }

        np.save(f"{self.path_outputs}/{self.epoch}_predictions.npy", outputs)

    def get_overall_metrics(self, dataset, over_all_outputs=False):
        scale = dataset.max_delta_forceXYZ
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
                outputs[key] = np.concatenate(outputs[key])

        forces_gt = outputs["delta_forces_gt"] * scale  # N
        forces_pred = outputs["delta_forces_pred"] * scale  # N
        slip_pred = outputs["slip_prob_pred"]
        slip_labels_pred = np.where(slip_pred[:, 1] > self.th, 1, 0)

        # smooth out labels
        slip_labels_gt = []
        slip_labels_pred_smooth = []
        n_trajectories = len(dataset.traj2idx) - 1

        for traj_id in range(1, n_trajectories):
            idxs = dataset.traj2idx[traj_id]
            t_slip_gt = outputs["slip_label_gt"][idxs]
            t_slip_pred = slip_labels_pred[idxs]

            # smooth slip prediction
            w = 3
            slip_window = deque([0] * w, maxlen=w)
            for idx in range(len(t_slip_pred)):
                slip_window.append(t_slip_pred[idx])
                t_slip_pred[idx] = 1 if sum(slip_window) == len(slip_window) else 0
            slip_labels_pred_smooth.append(t_slip_pred)
            slip_labels_gt.append(t_slip_gt)

        slip_labels_gt = np.concatenate(slip_labels_gt)
        slip_labels_pred = np.concatenate(slip_labels_pred_smooth)

        idx_shuffle = np.arange(len(slip_labels_gt))
        np.random.shuffle(idx_shuffle)
        slip_labels_gt = slip_labels_gt[idx_shuffle]
        slip_pred = slip_pred[idx_shuffle]
        slip_labels_pred = slip_labels_pred[idx_shuffle]

        report = classification_report(
            slip_labels_gt,
            slip_labels_pred,
            target_names=["no_slip", "slip"],
            output_dict=True,
        )
        b_acc = balanced_accuracy_score(slip_labels_gt, slip_labels_pred)

        rmse = np.sqrt((forces_gt - forces_pred) ** 2).mean(axis=0)
        rmse_std = np.sqrt((forces_gt - forces_pred) ** 2).std(axis=0)
        corr = np.array(
            [stats.pearsonr(forces_gt[:, i], forces_pred[:, i])[0] for i in range(3)]
        )

        metrics = {}
        for key in report["no_slip"].keys():
            metrics[f"no_slip/{key}"] = report["no_slip"][key]
        for key in report["slip"].keys():
            metrics[f"slip/{key}"] = report["slip"][key]
        metrics["balanced_accuracy"] = b_acc
        metrics["accuracy"] = report["accuracy"]
        metrics["delta_force/rmse"] = rmse
        metrics["delta_force/rmse_std"] = rmse_std
        metrics["delta_force/corr"] = corr
        metrics["n_samples"] = forces_gt.shape[0]

        np.save(f"{self.path_output_model}/{self.epoch}_metrics.npy", metrics)

    def make_plots(self, test_dset):
        outputs = np.load(
            f"{self.path_outputs}/{self.epoch}_predictions.npy", allow_pickle=True
        ).item()
        delta_forces_gt = outputs["delta_forces_gt"]
        delta_forces_pred = outputs["delta_forces_pred"]
        slip_prob_pred = outputs["slip_prob_pred"]
        slip_labels_pred = np.where(slip_prob_pred[:, 1] > self.th, 1, 0)

        n_trajectories = len(test_dset.traj2idx) - 1
        n_trajectories_plot = 20
        traj_ids_plot = np.linspace(1, n_trajectories, n_trajectories_plot, dtype=int)

        for traj_id in traj_ids_plot:
            idxs = test_dset.traj2idx[traj_id]
            t_slip_gt = outputs["slip_label_gt"][idxs]
            t_slip_pred = slip_labels_pred[idxs]
            t_delta_forces_gt = delta_forces_gt[idxs]
            t_delta_forces_pred = delta_forces_pred[idxs]

            self.plot_slip(test_dset, traj_id, t_slip_gt, t_slip_pred)
            self.plot_delta_forces(traj_id, t_delta_forces_gt, t_delta_forces_pred)

    def plot_slip(self, test_dset, traj_id, slip_gt, slip_pred):
        pose, force = test_dset._get_trajectory_pose_force(traj_id)
        horizon = test_dset.slip_horizon
        horizon_t = (horizon * 1 / 60.0) * 1000.0

        # smooth slip prediction
        w = 3
        slip_window = deque([0] * w, maxlen=w)
        for idx in range(len(slip_pred)):
            slip_window.append(slip_pred[idx])
            slip_pred[idx] = 1 if sum(slip_window) == len(slip_window) else 0

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        fig.tight_layout(pad=3.0)
        size_marker = 10

        colors = {"no_slip": "#369407", "slip": "#bb65fc", "error": "#fc0303"}

        # plot slip over time
        t = np.arange(len(slip_gt)) * 1 / 60.0
        axs[0].plot(
            t,
            slip_gt,
            color="blue",
            alpha=0.5,
            linewidth=5,
            label=(
                f"Ground truth (anywhere in next {horizon_t:.2f} ms)"
                if horizon > 0
                else "Ground truth"
            ),
        )
        axs[0].plot(
            t,
            slip_pred,
            color="red",
            label=(
                f"Prediction (anywhere in next {horizon_t:.2f} ms)"
                if horizon > 0
                else "Prediction"
            ),
        )
        axs[0].set_xlabel("t (s)")
        axs[0].set_ylim(-0.5, 1.5)
        axs[0].set_yticks([0, 1])
        axs[0].set_yticklabels(["No slip", "Slip"])
        axs[0].legend()
        axs[0].grid(True)
        axs[0].grid(which="minor", linestyle="--")

        idx_no_slip_agree = np.where((slip_gt == 0) & (slip_pred == 0))[0]
        idx_slip_agree = np.where((slip_gt == 1) & (slip_pred == 1))[0]
        idx_error = np.where(slip_gt != slip_pred)[0]

        if pose is not None:
            # plot displacement over time
            pX = pose[:, 0] - pose[0, 0]
            pY = pose[:, 1] - pose[0, 1]
            displ = np.array([((dx**2 + dy**2) ** 0.5) for dx, dy in zip(pX, pY)])

            axs[1].scatter(
                t[idx_no_slip_agree],
                displ[idx_no_slip_agree],
                c=colors["no_slip"],
                alpha=1.0,
                label=f"No slip (h={horizon})" if horizon > 0 else "No slip",
                s=size_marker,
            )
            axs[1].scatter(
                t[idx_slip_agree],
                displ[idx_slip_agree],
                c=colors["slip"],
                alpha=1.0,
                label=f"Slip (h={horizon})" if horizon > 0 else "Slip",
                s=size_marker,
            )
            axs[1].scatter(
                t[idx_error],
                displ[idx_error],
                c=colors["error"],
                label="Error",
                s=size_marker * 2,
            )
            axs[1].set_xlabel("Samples")
            axs[1].set_ylabel("Displacement")
            axs[1].legend()

        # plot friction cone
        mag_shear = (
            np.array(
                [((fx**2 + fy**2) ** 0.5) for fx, fy in zip(force[:, 0], force[:, 1])]
            )
            / 1000.0
        )
        mag_normal = (force[:, 2] * -1.0) / 1000.0

        coef_friction = test_dset.trajectories[traj_id]["coef_friction"]

        x = np.linspace(0, mag_shear.max() * 0.9, 100)
        y = 1 / coef_friction * x
        axs[2].plot(x, y, "--", c="gray", label="Friction Boundary")

        axs[2].scatter(
            mag_shear[idx_slip_agree],
            mag_normal[idx_slip_agree],
            c=colors["slip"],
            alpha=1.0,
            s=size_marker,
            label="Slip",
        )
        axs[2].scatter(
            mag_shear[idx_no_slip_agree],
            mag_normal[idx_no_slip_agree],
            c=colors["no_slip"],
            alpha=1.0,
            s=size_marker,
            label="No Slip",
        )
        axs[2].scatter(
            mag_shear[idx_error],
            mag_normal[idx_error],
            c=colors["error"],
            s=size_marker * 2,
            label="Error",
        )
        axs[2].set_xlabel("GT Shear Force (N)")
        axs[2].set_ylabel("GT Normal Force (N)")
        axs[2].legend()

        plt.savefig(f"{self.path_outputs}/{self.epoch}_{traj_id}_slip.png", dpi=300)
        plt.close()

    def plot_delta_forces(self, traj_id, delta_forces_gt, delta_forces_pred):
        fig, axs = plt.subplots(1, 2, figsize=(15, 4))
        fig.tight_layout(pad=3.0)

        delta_mag_shear_pred = np.array(
            [
                ((fx**2 + fy**2) ** 0.5)
                for fx, fy in zip(delta_forces_pred[:, 0], delta_forces_pred[:, 1])
            ]
        )
        delta_mag_normal_pred = delta_forces_pred[:, 2]

        delta_mag_shear_gt = np.array(
            [
                ((fx**2 + fy**2) ** 0.5)
                for fx, fy in zip(delta_forces_gt[:, 0], delta_forces_gt[:, 1])
            ]
        )

        delta_mag_normal_gt = delta_forces_gt[:, 2]

        t = np.arange(len(delta_mag_normal_gt)) * 1 / 60.0

        axs[0].plot(
            t,
            delta_mag_shear_gt,
            c="gray",
            linestyle="--",
            label="GT Δ shear",
        )
        axs[0].plot(
            t,
            delta_mag_shear_pred,
            c="blue",
            label="Pred Δ shear",
        )
        axs[0].set_xlabel("t (s)")
        axs[0].set_ylabel("$Δ Shear (N)")
        axs[0].legend()

        axs[1].plot(
            t,
            delta_mag_normal_gt,
            c="gray",
            linestyle="--",
            label="GT Δ normal",
        )
        axs[1].plot(
            t,
            delta_mag_normal_pred,
            c="green",
            label=f"Pred Δ normal",
        )
        axs[1].set_xlabel("t (s)")
        axs[1].set_ylabel("Δ Normal (N)")
        axs[1].legend()

        plt.savefig(f"{self.path_outputs}/{self.epoch}_{traj_id}_delta_forces.png", dpi=300)
        plt.close()