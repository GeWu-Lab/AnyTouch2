# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from .ssl_utils import warp, robost_loss, smooth_1st_loss


class SSL_loss(object):
    def __init__(self, config, frame_ids, ssim_loss=None):
        super().__init__()
        self.config = config
        self.frame_ids = frame_ids
        self.with_ssim = config["with_ssim"]
        if self.with_ssim:
            self.ssim = ssim_loss

    # normal force SSL loss
    def compute_losses_normal(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch"""
        losses = {}
        loss = 0
        reprojection_losses = []

        disp = outputs["normal"]
        color = inputs[:, 0:3, :, :]
        target = inputs[:, 0:3, :, :]

        for frame_id in self.frame_ids[1:]:
            pred = outputs[("color", frame_id)]
            reprojection_losses.append(self.compute_reprojection_loss(pred, target))

        reprojection_losses = torch.cat(reprojection_losses, 1)

        if reprojection_losses.shape[1] == 1:
            to_optimise = reprojection_losses
        else:
            to_optimise, idxs = torch.min(reprojection_losses, dim=1)

        to_optimise, idxs = torch.min(reprojection_losses, dim=1)
        loss += to_optimise.mean()
        losses["normal_reprojection_loss"] = to_optimise.mean()
        # print("loss: ", loss.item())

        mean_disp = disp.mean(2, True).mean(3, True)
        norm_disp = disp / (mean_disp + 1e-7)
        smooth_loss = self.get_smooth_loss(norm_disp, color)

        disparity_smoothness = float(self.config["disparity_smoothness"])
        smooth_loss = disparity_smoothness * smooth_loss  # / (2**scale)
        loss += smooth_loss
        losses["normal_smooth_loss"] = smooth_loss
        losses["normal_loss"] = loss * 5.0
        return losses

    def get_smooth_loss(self, disp, img):
        """Computes the smoothness loss for a disparity image
        The color image is used for edge-aware smoothness
        """
        grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

        grad_img_x = torch.mean(
            torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True
        )
        grad_img_y = torch.mean(
            torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True
        )

        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)

        return grad_disp_x.mean() + grad_disp_y.mean()

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images"""
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.with_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss
            # reprojection_loss = 0.45 * ssim_loss + 0.55 * l1_loss

        return reprojection_loss

    # shear force SSL loss

    def compute_losses_shear(self, inputs, outputs):
        """Compute the reprojection and smoothness optical flow losses for a minibatch"""
        losses = {}
        im0 = inputs[:, 0:3, :, :]
        im1 = inputs[:, 3:6, :, :]
        flow = outputs["shear"]
        im1_warp = warp(im0, flow)
        photometric_loss = robost_loss(im1, im1_warp)
        smooth_loss = smooth_1st_loss(flow=flow, image=im0)
        loss = photometric_loss + 0.05 * smooth_loss
        losses["shear_photometric_loss"] = photometric_loss
        losses["shear_smooth_loss"] = smooth_loss
        losses["shear_loss"] = loss
        return losses

    def compute_loss(self, inputs, outputs):
        losses = {}
        losses_normal = self.compute_losses_normal(inputs, outputs)
        losses_shear = self.compute_losses_shear(inputs, outputs)
        losses.update(losses_normal)
        losses.update(losses_shear)
        loss = losses["normal_loss"] + losses["shear_loss"]
        return loss, losses


# ==================== SSIM ====================
class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images"""

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01**2
        self.C2 = 0.03**2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x**2) - mu_x**2
        sigma_y = self.sig_y_pool(y**2) - mu_y**2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x**2 + mu_y**2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)
