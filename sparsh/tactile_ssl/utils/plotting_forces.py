# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import io
from matplotlib.axes import SubplotBase
from numpy.typing import NDArray
from functools import partial
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import numpy as np

colors = ["#7998e8", "#52a375", "#803b6b"]


def plot_correlation(forces_gt, forces_pred):
    corr_metric = []
    for i in range(3):
        corr = stats.pearsonr(forces_gt[:, i], forces_pred[:, i])
        corr_metric.append(corr[0])

    correlation_fig = plt.figure(figsize=(20, 5))
    axs: np.ndarray = correlation_fig.subplots(1, 3)
    for i, (force_gt, force_pred) in enumerate(zip(forces_gt.T, forces_pred.T)):
        axs[i].scatter(
            force_gt,
            force_pred,
            s=2,
            color=colors[i],
            label=f"r={corr_metric[i]:.3f}",
        )
        axs[i].set_xlabel("Ground Truth (N)")
        axs[i].set_ylabel("Prediction (N)")
        axs[i].set_title(f"Force {['X', 'Y', 'Z'][i]}")
        axs[i].grid(True)
        # plot 1:1 line
        axs[i].plot(
            [force_gt.min(), force_gt.max()],
            [force_gt.min(), force_gt.max()],
            "--",
            color="gray",
        )
        axs[i].legend()
    # return correlation_fig, axs

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format="png")
    plt.close("all")
    im = Image.open(img_buf)
    return im


def plot_forces_error(forces_gt, forces_pred, n_bins=100, n_std=3):
    forces_mse_xyz = (forces_pred - forces_gt) ** 2
    forces_rmse_xyz = np.sqrt(forces_mse_xyz) * 1000  # in mN
    error_fig = plt.figure(figsize=(20, 4))
    cone_fig = plt.figure(figsize=(20, 4))
    error_axs: np.ndarray = error_fig.subplots(1, 3)
    cone_axs: np.ndarray = cone_fig.subplots(1, 3)
    magnitude_force = [
        ((fx**2 + fy**2) ** 0.5) for fx, fy in zip(forces_gt[:, 0], forces_gt[:, 1])
    ]

    max_frmse = forces_rmse_xyz.mean(axis=0).max() + n_std * forces_rmse_xyz.std(axis=0).max()
    # plot magnitude of force vs force z and add color based on forces_mse
    labels = [r"$F_x$", r"$F_y$", r"$F_z$"]
    x_rmse = forces_rmse_xyz[:, 0].mean()
    y_rmse = forces_rmse_xyz[:, 1].mean()
    z_rmse = forces_rmse_xyz[:, 2].mean()
    total_rmse = x_rmse + y_rmse + z_rmse
    print(
        f"RMSE X: {x_rmse:.3f} mN, Y: {y_rmse:.3f} mN, Z: {z_rmse:.3f} mN, Total: {total_rmse:.3f} mN"
    )
    for i, (label, color) in enumerate(zip(labels, colors)):
        f_rmse = forces_rmse_xyz[:, i]
        sc = cone_axs[i].scatter(
            magnitude_force,
            forces_gt[:, 2],
            c=f_rmse,
            cmap="viridis",
            vmin=0,
            # vmax=f_mse.mean() + n_std * f_mse.std(),
            vmax=max_frmse,
            marker="o",
            s=2,
        )
        cbar = plt.colorbar(sc, ax=cone_axs[i])
        cbar.set_label(f"Error {label} (mN)")
        cone_axs[i].set_xlabel("Tangential Force")
        cone_axs[i].set_ylabel("Normal Force")
        cone_axs[i].set_title(f"RMSE {label}")
        cone_axs[i].grid(True)

        # plot distribution of f_mse
        error_axs[i].hist(
            f_rmse,
            bins=n_bins,
            alpha=0.9,
            color=color,
            edgecolor="gray",
            label=f"n={f_rmse.shape[0]} \n mean={f_rmse.mean():.3f}mN \n std={f_rmse.std():.3f}mN",
        )
        # x axis only mean + 3std
        error_axs[i].set_xlim(0, f_rmse.mean() + n_std * f_rmse.std())
        error_axs[i].set_xlabel(f"RMSE {label} (mN)")
        error_axs[i].set_ylabel("Frequency")
        error_axs[i].set_title(f"Distribution of RMSE {label}")
        error_axs[i].legend()
        error_axs[i].grid(True)
    error_fig.suptitle(f" RMSE mean ± {n_std}std")

    img_buf = io.BytesIO()
    error_fig.savefig(img_buf, format="png")
    im_error = Image.open(img_buf)
    img_buf = io.BytesIO()
    cone_fig.savefig(img_buf, format="png")
    im_cone = Image.open(img_buf)
    plt.close("all")
    return im_error, im_cone
