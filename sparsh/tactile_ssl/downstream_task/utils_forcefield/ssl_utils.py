# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import io
from typing import Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from torch import Tensor, nn
from torch.autograd import Variable


def digit_intrinsics():
    yfov = 60
    # W, H = 240, 320
    W, H = 224, 224
    fx = H * 0.5 / np.tan(np.deg2rad(yfov) * 0.5)
    fy = fx
    cx, cy = W / 2.0, H / 2.0
    K = np.array([[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    # K[0, :] /= W
    # K[1, :] /= H
    invK = np.linalg.inv(K)
    K = torch.tensor(K, dtype=torch.float32)
    invK = torch.tensor(invK, dtype=torch.float32)
    return K, invK


def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow

    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow

    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = Variable(grid) + flo

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = nn.functional.grid_sample(x, vgrid)
    mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
    mask = nn.functional.grid_sample(mask, vgrid)

    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1

    return output * mask


def robost_loss(im, im_warp, p=2, q=None, eps=None):
    # Copyright (c) OpenMMLab. All rights reserved.
    epe_map = torch.norm(im - im_warp, p, dim=1)
    if q is not None and eps is not None:
        epe_map = (epe_map + eps) ** q
    return epe_map.mean()


def gradient(data: Tensor, stride: int = 1) -> Tuple[Tensor]:
    # https://github.com/open-mmlab/mmflow/blob/9fb1d2f1bb3de641ddcba0dd355064b6ed9419f4/mmflow/models/losses/smooth_loss.py#L27
    """Calculate gradient in data.

    Args:
        data (Tensor): input data with shape (B, C, H, W).
        stride (int): stride for distance of calculating changing. Default to
            1.

    Returns:
        tuple(Tensor): partial derivative of data with respect to x, with shape
            (B, C, H-stride, W), and partial derivative of data with respect to
            y with shape (B, C, H, W-stride).
    """
    D_dy = data[:, :, stride:] - data[:, :, :-stride]
    D_dx = data[:, :, :, stride:] - data[:, :, :, :-stride]
    return D_dx / stride, D_dy / stride


def smooth_1st_loss(
    flow: Tensor,
    image: Tensor,
    alpha: float = 0.0,
    smooth_edge_weighting: str = "exponential",
) -> Tensor:
    """The First order smoothness loss.

    Modified from
    https://github.com/lliuz/ARFlow/blob/master/losses/flow_loss.py
    licensed under MIT License,
    and https://github.com/google-research/google-research/blob/master/uflow/uflow_utils.py
    licensed under the Apache License, Version 2.0.

    Args:
        flow (Tensor): Input optical flow with shape (B, 2, H, W).
        image (Tensor): Input image with shape (B, 3, H, W).
        alpha (float): Weight for modulates edge weighting. Default to: 0.
        smooth_edge_weighting (str): Function for calculating abstract
            value of image gradient which can be a string {'exponential'
            'gaussian'}.

    Returns:
        Tensor: A scaler of the first order smoothness loss.
    """  # noqa E501
    assert smooth_edge_weighting in ("exponential", "gaussian"), (
        "smooth edge function must be `exponential` or `gaussian`,"
        f"but got {smooth_edge_weighting}"
    )
    # Compute image gradients and sum them up to match the receptive field
    img_dx, img_dy = gradient(image)

    abs_fn = None
    if smooth_edge_weighting == "gaussian":
        abs_fn = torch.square
    elif smooth_edge_weighting == "exponential":
        abs_fn = torch.abs

    weights_x = torch.exp(-torch.mean(abs_fn(img_dx * alpha), 1, keepdim=True))
    weights_y = torch.exp(-torch.mean(abs_fn(img_dy * alpha), 1, keepdim=True))

    dx, dy = gradient(flow)

    loss_x = weights_x * dx.abs() / 2.0
    loss_y = weights_y * dy.abs() / 2.0

    return loss_x.mean() + loss_y.mean()


# ==================== Depth reprojection ====================
class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud"""

    def __init__(self, height, width):
        super(BackprojectDepth, self).__init__()

        # self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing="xy")
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(
            torch.from_numpy(self.id_coords), requires_grad=False
        )
        self.pix_coords = torch.unsqueeze(
            torch.stack([self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0
        )

    def forward(self, depth, inv_K):
        batch_size = depth.size(0)

        ones = torch.ones(batch_size, 1, self.height * self.width).to(depth.device)
        pix_coords = self.pix_coords.repeat(batch_size, 1, 1).to(depth.device)
        pix_coords = torch.cat([pix_coords, ones], 1)

        # repreat inv_K for batch_size times
        inv_K = inv_K.unsqueeze(0).repeat(batch_size, 1, 1).to(depth.device)

        cam_points = torch.matmul(inv_K[:, :3, :3], pix_coords)
        cam_points = depth.view(batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, ones], 1)

        return cam_points


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T"""

    def __init__(self, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        # self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        batch_size = points.size(0)
        K = K.unsqueeze(0).repeat(batch_size, 1, 1).to(points.device)
        P = torch.matmul(K, T)[:, :3, :]

        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (
            cam_points[:, 2, :].unsqueeze(1) + self.eps
        )
        pix_coords = pix_coords.view(batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords


# ==================== Other functions ====================
def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1.0 / max_depth
    max_disp = 1.0 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1.0 / scaled_disp
    return scaled_disp, depth


def plot_quiver(shear, normal, i_sample, spacing, margin=0, **kwargs):
    """Plots less dense quiver field.

    Args:
        ax: Matplotlib axis
        flow: motion vectors
        spacing: space (px) between each arrow in grid
        margin: width (px) of enclosing region without arrows
        kwargs: quiver kwargs (default: angles="xy", scale_units="xy")
    """
    fig, ax = plt.subplots()
    h, w, *_ = shear.shape

    nx = int((w - 2 * margin) / spacing)
    ny = int((h - 2 * margin) / spacing)

    x = np.linspace(margin, w - margin - 1, nx, dtype=np.int64)
    y = np.linspace(margin, h - margin - 1, ny, dtype=np.int64)

    shear = shear[np.ix_(y, x)]
    u = shear[:, :, 0]
    v = shear[:, :, 1]
    m = normal[np.ix_(y, x)]

    # rad = np.sqrt(np.square(u) + np.square(v))
    # rad_max = np.max(rad)
    rad_max = 20.0
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)

    u = np.clip(u, -1.0, 1.0)
    v = np.clip(v, -1.0, 1.0)
    uu = u.copy()
    vv = v.copy()
    r = np.sqrt(u**2 + v**2)
    idx_clip = np.where(r < 0.01)
    uu[idx_clip] = 0.0
    vv[idx_clip] = 0.0

    uu = uu / (np.abs(uu).max() + epsilon)
    vv = vv / (np.abs(vv).max() + epsilon)

    kwargs = {
        **dict(
            angles="uv",
            scale_units="dots",
            scale=0.025,
            width=0.007,
            cmap="inferno",
            edgecolor="face",
        ),
        **kwargs,
    }

    ax_shear = ax.quiver(y, x, uu, -vv, m, **kwargs)

    ax.set_ylim(sorted(ax.get_ylim(), reverse=True))
    ax.set_facecolor("black")
    ax.set_xticks([])
    ax.set_yticks([])

    with io.BytesIO() as buff:
        fig.savefig(
            buff, format="png", bbox_inches="tight", pad_inches=0, transparent=False
        )
        buff.seek(0)
        img = Image.open(io.BytesIO(buff.read()))
    plt.close(fig)
    return np.array(img)


def plot_quiver_img(img, shear, normal, mask, spacing, margin=0, **kwargs):
    """Plots less dense quiver field.

    Args:
        ax: Matplotlib axis
        flow: motion vectors
        spacing: space (px) between each arrow in grid
        margin: width (px) of enclosing region without arrows
        kwargs: quiver kwargs (default: angles="xy", scale_units="xy")
    """
    fig, ax = plt.subplots()
    h, w, *_ = shear.shape

    nx = int((w - 2 * margin) / spacing)
    ny = int((h - 2 * margin) / spacing)

    x = np.linspace(margin, w - margin - 1, nx, dtype=np.int64)
    y = np.linspace(margin, h - margin - 1, ny, dtype=np.int64)

    shear = shear[np.ix_(y, x)]
    mask = mask[np.ix_(y, x)]
    u = shear[:, :, 0]  # * mask
    v = shear[:, :, 1]  # * mask
    m = normal[np.ix_(y, x)]

    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = 100.0
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)

    kwargs = {
        **dict(
            angles="xy",
            scale_units="xy",
            cmap="gnuplot",
            width=0.005,
            clim=(0, 1),
        ),
        **kwargs,
    }

    ax.imshow(img)
    ax.quiver(x, y, u, v, m, **kwargs)

    ax.set_ylim(sorted(ax.get_ylim(), reverse=True))
    ax.set_aspect("equal")

    ax.set_facecolor("black")
    ax.set_xticks([])
    ax.set_yticks([])

    with io.BytesIO() as buff:
        fig.savefig(
            buff, format="png", bbox_inches="tight", pad_inches=0, transparent=False
        )
        buff.seek(0)
        img = Image.open(io.BytesIO(buff.read()))
    plt.close(fig)
    return img
