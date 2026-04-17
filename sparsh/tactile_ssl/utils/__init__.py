# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

from typing import Literal, List, Tuple
import os
import torch
import einops


def get_local_rank() -> int:
    if "LOCAL_RANK" in os.environ:
        local_rank = os.environ["LOCAL_RANK"]
    elif "SLURM_JOB_ID" in os.environ:
        try:
            local_rank = os.environ["SLURM_LOCALID"]
        except:
            local_rank = 0
    else:
        local_rank = 0
    return local_rank


def apply_masks(x, masks, concat=True):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param masks: list of tensors containing indices of patches in [N] to keep
    """
    all_x = []
    for i, mask in enumerate(masks):
        mask_keep = einops.repeat(mask, "b n -> b n d", d=x.size(-1))
        all_x.append(torch.gather(x, dim=-2, index=mask_keep))
    if not concat: 
        return all_x
    return torch.cat(all_x, dim=0)


def create_ndgrid(
    resolution: List[int],
    device: torch.device = torch.device("cpu"),
    normalized_coords: bool = True,
    indexing: Literal["xy", "ij"] = "ij",
    dtype: torch.dtype = torch.float,
) -> torch.Tensor:
    """
    Create a n-D grid with coordinates in range [-1, 1]
    **supports only upto 3D**
    Args:
        resolution: Resolution of the grid
        device: Device to create the grid on
        normalized_coords: If True, the grid will be in range [-1, 1]
        indexing: Indexing mode of the grid
        dtype: Data type of the grid
    """
    assert len(resolution) <= 3, "Only upto 3D grids are supported"
    axes = []
    if normalized_coords:
        for res in resolution:
            axes.append(torch.linspace(-1, 1, res + 1, dtype=dtype, device=device)[:-1])
    else:
        for res in resolution:
            axes.append(torch.arange(0, res, dtype=dtype, device=device))
    grid = torch.stack(torch.meshgrid(*axes, indexing=indexing), dim=-1)
    if len(resolution) == 2:
        grid = einops.rearrange(grid, "y x ... -> (y x) ...")
    elif len(resolution) == 3:
        grid = einops.rearrange(grid, "z y x ... -> (z y x) ...")
    return grid


def quaternion_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part last.
        b: Quaternions as tensor of shape (..., 4), real part last.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    ax, ay, az, aw = torch.unbind(a, -1)
    bx, by, bz, bw = torch.unbind(b, -1)
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    ow = aw * bw - ax * bx - ay * by - az * bz
    return torch.stack((ox, oy, oz, ow), -1)


def quaternion_invert(quaternion: torch.Tensor) -> torch.Tensor:
    """
    Given a quaternion representing rotation, get the quaternion representing
    its inverse.

    Args:
        quaternion: Quaternions as tensor of shape (..., 4), with real part
            last, which must be versors (unit quaternions).

    Returns:
        The inverse, a tensor of quaternions of shape (..., 4).
    """

    scaling = torch.tensor([-1, -1, -1, 1], device=quaternion.device)
    return quaternion * scaling


def quaternion_apply(quaternion: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    """
    Apply the rotation given by a quaternion to a 3D point.
    Usual torch rules for broadcasting apply.

    Args:
        quaternion: Tensor of quaternions, real part first, of shape (..., 4).
        point: Tensor of 3D points of shape (..., 3).

    Returns:
        Tensor of rotated points of shape (..., 3).
    """
    if point.size(-1) != 3:
        raise ValueError(f"Points are not in 3D, {point.shape}.")

    real_parts = point.new_zeros(point.shape[:-1] + (1,))
    point_as_quaternion = torch.cat((point, real_parts), -1)
    out = quaternion_raw_multiply(
        quaternion_raw_multiply(quaternion, point_as_quaternion),
        quaternion_invert(quaternion),
    )
    return out[..., :-1]


def patchify_image(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Patchify an image into patches of a given size and stride.

    Args:
        x: Image tensor of shape (B, C, H, W).
        patch_size: Size of the patches.

    Returns:
        A tensor of shape (B, N, C, patch_size, patch_size), where N is the
        number of patches.
    """
    B, _, H, W = x.shape
    assert (
        H % patch_size == 0 and W % patch_size == 0
    ), "Image dimensions must be divisible by patch size."
    x = einops.rearrange(
        x, "b c (h p1) (w p2) -> b (h w) c p1 p2", p1=patch_size, p2=patch_size
    )
    return einops.rearrange(x, "b (h w) c p1 p2 -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size, h=H // patch_size, w=W//patch_size)


def patches_to_image(
    x: torch.Tensor,
    patch_size: int,
    image_size: Tuple[int, int]
) -> torch.Tensor:
    """
    Reconstruct an image from patches of a given size and stride.

    Args:
        x: Patch tensor of shape (B, patch_size * patch_size, C).
        patch_size: Size of the patches.
        image_size: Tuple[int, int] height, width
        stride: Stride of the patches.

    Returns:
        A tensor of shape (B, C, H, W), the reconstructed image.
    """
    B, P, _ = x.shape
    H, W = image_size
    assert (
        H % patch_size == 0 and W % patch_size == 0
    ), "Image dimensions must be divisible by patch size."
    n_h, n_w = H // patch_size, W // patch_size
    assert P == n_h * n_w, "Invalid number of patches in x"
    imgs = einops.rearrange(
        x,
        " b (h w)  (p q c) -> b c (h p) (w q)",
        p=patch_size,
        q=patch_size,
        h=n_h,
        w=n_w,
    )
    return imgs


def _get_conv1d_output_size(input_size, kernel_size, stride, padding, dilation):
    return (input_size - dilation * (kernel_size - 1) - 1 + 2 * padding) // stride + 1

class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.max = float("-inf")
        self.min = float("inf")
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        try:
            self.max = max(val, self.max)
            self.min = min(val, self.min)
        except Exception:
            pass
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count