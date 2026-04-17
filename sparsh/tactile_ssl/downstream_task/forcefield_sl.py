from typing import Any, Dict, Tuple, Optional, List, Union
from matplotlib.axes import SubplotBase
from numpy.typing import NDArray
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import flow_to_image

from sparsh.tactile_ssl.utils.logging import get_pylogger
from sparsh.tactile_ssl.downstream_task.sl_module import SLModule

from sparsh.tactile_ssl.downstream_task.utils_forcefield.layers.Fusion import Fusion
from sparsh.tactile_ssl.downstream_task.utils_forcefield.layers.Reassemble import Reassemble
from sparsh.tactile_ssl.downstream_task.utils_forcefield.layers.Head import NormalShearHead
from sparsh.tactile_ssl.downstream_task.utils_forcefield.pose_estimator.PoseEstimator import (
    PoseEstimator,
)
from sparsh.tactile_ssl.downstream_task.utils_forcefield.ssl_flow_loss import SSL_loss, SSIM
from sparsh.tactile_ssl.downstream_task.utils_forcefield.ssl_utils import (
    BackprojectDepth,
    Project3D,
    disp_to_depth,
    digit_intrinsics,
)
from sparsh.tactile_ssl.model import VIT_EMBED_DIMS

log = get_pylogger(__name__)


class ForceFieldDecoder(nn.Module):
    def __init__(
        self,
        image_size=(3, 224, 224),
        embed_dim='base',
        patch_size=16,
        resample_dim=128,
        norm_layer=nn.LayerNorm,
        hooks=[2, 5, 8, 11],
        reassemble_s=[4, 8, 16, 32],
    ):
        super().__init__()

        embed_dim = VIT_EMBED_DIMS[f"vit_{embed_dim}"]
        self.norm = norm_layer(embed_dim)
        self.hooks = hooks
        self.n_patches = (image_size[1] // patch_size) ** 2

        # Reassembles Fusion
        self.reassembles = []
        self.fusions = []
        for s in reassemble_s:
            self.reassembles.append(
                Reassemble(image_size, "ignore", patch_size, s, embed_dim, resample_dim)
            )
            self.fusions.append(Fusion(resample_dim))
        self.reassembles = nn.ModuleList(self.reassembles)
        self.fusions = nn.ModuleList(self.fusions)

        # head to decode force field
        self.probe = NormalShearHead(features=resample_dim)

    def forward(self, encoder_activations, mode="normal_shear"):
        sample_key = list(encoder_activations.keys())[0]
        start_idx = encoder_activations[sample_key].shape[1] - self.n_patches
        for b in encoder_activations.keys():
            encoder_activations[b] = self.norm(encoder_activations[b][:, start_idx:, :])

        previous_stage = None
        for i in np.arange(len(self.fusions) - 1, -1, -1, dtype=int):
            hook_to_take = "t" + str(self.hooks[int(i)])
            activation_result = encoder_activations[hook_to_take]
            reassemble_result = self.reassembles[i](activation_result)
            fusion_result = self.fusions[i](reassemble_result, previous_stage)
            previous_stage = fusion_result

        y = self.probe(previous_stage, mode)

        outputs = {}
        if mode == "normal_shear":
            outputs["normal"] = y[:, 0, :, :].unsqueeze(1)
            outputs["shear"] = y[:, 1:, :, :]
        elif mode == "normal":
            outputs["normal"] = y
        elif mode == "shear":
            outputs["shear"] = y
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        return outputs


class ForceFieldModule(SLModule):
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
        ssl_config: Dict[str, Any] = {},
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

        self.ssl_config = ssl_config
        self.with_sl_supervision = ssl_config["loss"]["with_sl_supervision"]
        self.with_mask_supervision = ssl_config["loss"]["with_mask_supervision"]
        self.frame_ids = [0, -1]
        img_sz = ssl_config["img_sz"]
        self.device = torch.device("cuda")
        self.pose_estimator = PoseEstimator(
            num_encoder_layers=ssl_config["pose_estimator"]["num_encoder_layers"],
            frame_ids=self.frame_ids,
        )

        # view synthesis
        self.backproject_depth = BackprojectDepth(img_sz[0], img_sz[1])
        self.project_3d = Project3D(img_sz[0], img_sz[1])

        # force SSL loss
        self.ssim = None
        if ssl_config["loss"]["with_ssim"]:
            self.ssim = SSIM()
        self.ssl_loss_fn = SSL_loss(
            ssl_config["loss"], self.frame_ids, ssim_loss=self.ssim
        )

        self.k, self.inv_k = digit_intrinsics()

        self.val_input_tactile = []
        self.val_output_tactile = []
        self.val_output_normal = []
        self.val_output_shear = []

        print("ForceFieldModule initialized")

        self.hooks = [2, 5, 8, 11]
        self.register_hooks()

    def register_hooks(self):
        self.encoder_activations = {}
        self._get_layers_from_hooks()

    def _get_layers_from_hooks(self):
        def get_activation(name):
            def hook(model, input, output):
                self.encoder_activations[name] = output

            return hook

        for h in self.hooks:
            self.model_encoder.blocks[h].register_forward_hook(
                get_activation("t" + str(h))
            )

    def forward(self, x: torch.Tensor, mode="normal_shear") -> torch.Tensor:
        z = self.model_encoder(x)
        if self.train_encoder:
            for k in self.encoder_activations.keys():
                self.encoder_activations[k] = self.encoder_activations[k].detach()
        y_pred = self.model_task(self.encoder_activations, mode)
        return y_pred

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        outputs = {'normal': None, 'shear': None}


        # forward pass for normal
        x = batch["image_bg"]
        outputs_normal = self.forward(x, mode="normal")
        outputs['normal'] = outputs_normal['normal']

        # forward pass for shear
        x = batch["image"]
        output_poses = self.pose_estimator(x)
        outputs_shear = self.forward(x, mode="shear")
        outputs['shear'] = outputs_shear['shear']

        outputs.update(output_poses)
        # outputs.update(outputs_forces)

        # apply reprojection
        self.generate_images_pred(x, outputs)
        # compute loss
        loss, losses = self.ssl_loss_fn.compute_loss(x, outputs)

        outputs["loss"] = loss
        outputs["normal_loss"] = losses["normal_loss"]
        outputs["shear_loss"] = losses["shear_loss"]

        if self.with_mask_supervision:
            mask = batch["mask"][:, None, :, :]
            normal = outputs["normal"]
            normal_loss = F.smooth_l1_loss(normal, mask * normal)
            outputs["loss"] += normal_loss  # + shear_loss
            outputs["normal_loss"] += normal_loss

        if self.with_sl_supervision:
            y_gt = batch["force"]
            mask = batch["mask"]
            y_pred = self.compute_sl_force(outputs, mask)
            loss_sl = F.smooth_l1_loss(y_pred, y_gt)
            mse_xyz = F.mse_loss(y_pred.detach(), y_gt.detach(), reduction="none").mean(
                dim=0
            )
            outputs["rmse_xyz"] = torch.sqrt(mse_xyz)
            outputs["loss"] += loss_sl

        return outputs

    def compute_sl_force(self, outputs, mask):
        img_sz = self.ssl_config["img_sz"]
        normal_unmask = outputs["normal"].squeeze(1)
        shear = outputs["shear"]
        f_z = normal_unmask  # * mask
        f_x = shear[:, 0, :, :]  # * mask
        f_y = shear[:, 1, :, :]  # * mask

        # sum over the batch
        f_x = f_x.sum(dim=[1, 2]) / (img_sz * img_sz)
        f_y = f_y.sum(dim=[1, 2]) / (img_sz * img_sz)
        f_z = f_z.sum(dim=[1, 2]) / (img_sz * img_sz)

        y_pred = torch.stack([f_x, f_y, f_z], dim=1)

        return y_pred

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        scale = 0
        source_scale = 0

        shear = outputs["shear"].float()
        outputs["flow"] = flow_to_image(shear)

        disp = outputs["normal"]
        min_depth = self.ssl_config["loss"]["min_depth"]
        max_depth = self.ssl_config["loss"]["max_depth"]
        _, depth = disp_to_depth(disp, min_depth, max_depth)

        for i, frame_id in enumerate(self.frame_ids[1:]):
            T = outputs[("cam_T_cam", frame_id)].float()
            cam_points = self.backproject_depth(depth, self.inv_k)
            pix_coords = self.project_3d(cam_points, self.k, T)

            outputs[("sample", frame_id)] = pix_coords

            outputs[("color", frame_id)] = F.grid_sample(
                inputs[:, 3:6, :, :],
                outputs[("sample", frame_id)],
                padding_mode="border",
                align_corners=True,
            )

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
            trainer_instance.wandb.log(
                {
                    f"{label}/normal_loss": outputs["normal_loss"],
                    f"global_{label}_step": step,
                }
            )
            trainer_instance.wandb.log(
                {
                    f"{label}/shear_loss": outputs["shear_loss"],
                    f"global_{label}_step": step,
                }
            )

            if self.with_sl_supervision:
                trainer_instance.wandb.log(
                    {
                        f"{label}/rmse_Fx": outputs["rmse_xyz"][0],
                        f"global_{label}_step": step,
                    }
                )
                trainer_instance.wandb.log(
                    {
                        f"{label}/rmse_Fy": outputs["rmse_xyz"][1],
                        f"global_{label}_step": step,
                    }
                )
                trainer_instance.wandb.log(
                    {
                        f"{label}/rmse_Fz": outputs["rmse_xyz"][2],
                        f"global_{label}_step": step,
                    }
                )

    def on_train_batch_end(self, outputs, batch, batch_idx, trainer_instance=None):
        self.log_metrics(outputs, trainer_instance.global_step, trainer_instance)

    def on_validation_batch_end(
        self, outputs: Dict, batch: Dict, batch_idx: int, trainer_instance=None
    ):
        if batch_idx % 10 == 0:
            n = 5
            batch_size = batch["image"].shape[0]
            idxs = np.random.choice(batch_size, n, replace=False)

            self.val_input_tactile.append(batch["image"][idxs, 3:6, :, :])
            self.val_output_tactile.append(outputs[("color", -1)][idxs])
            self.val_output_normal.append(outputs["normal"][idxs])
            self.val_output_shear.append(outputs["flow"][idxs])

        self.log_metrics(
            outputs, trainer_instance.global_val_step, trainer_instance, "val"
        )

    def on_validation_epoch_end(self, trainer_instance=None):

        def normalize_image(x):
            """Rescale image pixels to span range [0, 1]"""
            ma = float(x.max().cpu().data)
            mi = float(x.min().cpu().data)
            d = ma - mi if ma != mi else 1e5
            return (x - mi) / d

        self.val_input_tactile = torch.cat(self.val_input_tactile, dim=0)
        self.val_output_tactile = torch.cat(self.val_output_tactile, dim=0)
        self.val_output_normal = torch.cat(self.val_output_normal, dim=0)
        self.val_output_shear = torch.cat(self.val_output_shear, dim=0)

        # select nb_to_show images to show
        nb_to_show = 10
        idxs = np.random.choice(
            self.val_input_tactile.shape[0], nb_to_show, replace=False
        )

        x = self.val_input_tactile[idxs]
        tmp = x.permute(0, 2, 3, 1).detach().cpu().numpy()
        color = (tmp - tmp.min()) / (tmp.max() - tmp.min())

        x = self.val_output_tactile[idxs]
        tmp = x.permute(0, 2, 3, 1).detach().cpu().numpy()
        color_pred = (tmp - tmp.min()) / (tmp.max() - tmp.min())

        x = self.val_output_normal[idxs]
        x = normalize_image(x)
        x = x.repeat(1, 3, 1, 1)
        normal = x.permute(0, 2, 3, 1).detach().cpu().numpy()

        x = self.val_output_shear[idxs]
        tmp = x.permute(0, 2, 3, 1).detach().cpu().numpy()
        shear = tmp / 255.0

        if trainer_instance is not None:
            trainer_instance.wandb.log(
                {
                    f"val/org_color": [
                        trainer_instance.wandb.Image(im, caption="img_{}".format(i + 1))
                        for i, im in enumerate(color)
                    ]
                }
            )

            trainer_instance.wandb.log(
                {
                    f"val/pred_color": [
                        trainer_instance.wandb.Image(im, caption="img_{}".format(i + 1))
                        for i, im in enumerate(color_pred)
                    ]
                }
            )

            trainer_instance.wandb.log(
                {
                    f"val/normal": [
                        trainer_instance.wandb.Image(im, caption="img_{}".format(i + 1))
                        for i, im in enumerate(normal)
                    ]
                }
            )

            trainer_instance.wandb.log(
                {
                    f"val/shear": [
                        trainer_instance.wandb.Image(im, caption="img_{}".format(i + 1))
                        for i, im in enumerate(shear)
                    ]
                }
            )

        self.val_input_tactile = []
        self.val_output_tactile = []
        self.val_output_normal = []
        self.val_output_shear = []
