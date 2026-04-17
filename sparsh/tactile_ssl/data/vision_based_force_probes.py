# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

from omegaconf import DictConfig
import numpy as np

import torch
from torch.utils import data
import torchvision.transforms.functional as TF
from torchvision import transforms
import matplotlib.pyplot as plt

from sparsh.tactile_ssl.data.digit.utils import (
    load_dataset_forces,
    load_sample_from_buf,
    get_resize_transform,
    load_bin_image,
)
from sparsh.tactile_ssl.utils.logging import get_pylogger

log = get_pylogger()

DEBUG = False

class ForceDataset(data.Dataset):
    def __init__(
        self,
        config: DictConfig,
        dataset_name: str,
    ):
        super().__init__()
        self.config = config
        self.dataset_name = dataset_name

        self.remove_bg = (
            self.config.remove_bg if hasattr(self.config, "remove_bg") else False
        )
        self.out_format = self.config.out_format  # if output video
        assert self.out_format in [
            "video",
            "concat_ch_img",
            "single_image",
        ], ValueError(
            "out_format should be 'video' or 'concat_ch_img' or 'single_image'"
        )

        frame_stride = self.config.frame_stride
        self.num_frames = (
            1 if self.out_format == "single_image" else self.config.num_frames
        )
        self.frames_concat_idx = np.arange(
            0, self.num_frames * frame_stride, frame_stride
        )

        # load dataset
        self.frames, dataset_forces, _ = load_dataset_forces(
            self.config, dataset_name, config.sensor
        )
        # force and other labels
        self.dataset_force = dataset_forces["force"]

        # transforms
        self.img_sz = self.config.transforms.resize
        self.transform_resize = get_resize_transform(self.img_sz)

        # background
        self.bg = None
        no_contact_normal_force_threshold = 50
        if self.remove_bg:
            idx_bg = np.where(np.abs(self.dataset_force[:, 2]) < no_contact_normal_force_threshold)[0][0]
            self.bg = load_bin_image(self.frames[idx_bg])

        # force scale
        if "sphere" in dataset_name:
            self.max_abs_forceXYZ = self.config.sphere_max_abs_forceXYZ
        elif "sharp" in dataset_name:
            self.max_abs_forceXYZ = self.config.sharp_max_abs_forceXYZ
        elif "hex" in dataset_name:
            self.max_abs_forceXYZ = self.config.hex_max_abs_forceXYZ
        else:
            log.warn("Using default max_abs_forceXYZ, since probe name not found.")
        self.max_abs_forceXYZ = np.array(self.max_abs_forceXYZ)

    def __len__(self):
        return len(self.frames)

    def _plot_tactile_clip(self, clip):
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, self.num_frames, figsize=(20, 5))
        for i in range(self.num_frames):
            axs[i].imshow(clip[i].permute(1, 2, 0))
            axs[i].axis("off")
        plt.savefig("tactile_clip.png")

    def __getitem__(self, idx):
        try:
            inputs = {}
            inputs["image"] = self._get_tactile_images(idx)
            inputs["force"] = self._get_force_labels(idx)
            inputs["force_scale"] = torch.tensor(self.max_abs_forceXYZ, dtype=torch.float32)
            return inputs

        except Exception as e:
            log.warn(f"Error in loading sample {idx}: {e}. Resampling")
            self.__getitem__(np.random.randint(0, self.__len__()))

    def _get_tactile_images(self, idx_sample):

        sample_images = []
        for i in self.frames_concat_idx:
            frame_sample = np.clip(idx_sample - i, 0, len(self.frames) - 1)
            image = load_sample_from_buf(self.frames[frame_sample], self.bg)
            image = self.transform_resize(image)
            sample_images.append(image)

        if DEBUG:
            self._plot_tactile_clip(sample_images)

        if self.out_format == "single_image":
            output = sample_images[0]
        elif self.out_format == "video":
            output = torch.stack(sample_images, dim=0)
            output = output.permute(1, 0, 2, 3)
        elif self.out_format == "concat_ch_img":
            output = torch.cat(sample_images, dim=0)
        return output

    def _get_force_labels(self, idx_sample):
        force_gt = self.dataset_force[idx_sample] / 1000.0
        fx = force_gt[0]
        fy = force_gt[1]
        fz = force_gt[2] * -1.0
        fz = np.clip(fz, 0.0, 10.0)
        force_gt = np.array([fx, fy, fz]) / self.max_abs_forceXYZ
        force_gt = np.clip(force_gt, -1.0, 1.0)
        force_gt = torch.tensor(force_gt, dtype=torch.float32)
        return force_gt

class DigitForceDataset(ForceDataset):
   def __init__( 
        self,
        config: DictConfig,
        dataset_name: str,
    ):
        super().__init__(config, dataset_name)

class GelsightForceDataset(ForceDataset):
    def __init__(
        self,
        config: DictConfig,
        dataset_name: str,
    ):
        super().__init__(config, dataset_name)