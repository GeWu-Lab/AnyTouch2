# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

from typing import Literal
from omegaconf import DictConfig
import numpy as np
import random

import torch
from torch.utils import data
import torchvision.transforms.functional as TF
from torchvision import transforms

from sparsh.tactile_ssl.data.digit.utils import (
    load_pickle_dataset,
    load_sample_from_buf,
    get_resize_transform,
    get_bg_img,
)

from sparsh.tactile_ssl.utils.logging import get_pylogger
import matplotlib.pyplot as plt

log = get_pylogger()
DEBUG = False

class VisionTactileBackboneDataset(data.Dataset):
    def __init__(
        self,
        config: DictConfig,
        dataset_name: str,
        sensor_type: Literal["digit", "gelsight", "gelsight_mini"] = "digit",
    ):
        super().__init__()
        self.config = config
        self.sensor_type = self.config.sensor
        self.dataset_name = dataset_name

        assert self.sensor_type in ["digit", "gelsight", "gelsight_mini"], ValueError(
            "sensor_type should be 'digit', 'gelsight' or 'gelsight_mini'"
        )

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
        self.num_frames = 1 if self.out_format == "single_image" else self.config.num_frames
        self.frames_concat_idx = np.arange(0, self.num_frames * frame_stride, frame_stride)

        # load dataset
        file_dataset = f"{self.config.path_dataset}/{dataset_name}.pkl"
        self.frames = load_pickle_dataset(file_dataset)
        self.total_frames = len(self.frames)
        self.loader = load_sample_from_buf

        # background
        self.bg = get_bg_img(self.config, self.sensor_type, dataset_name, self.remove_bg)
              
        # transforms
        self.img_sz = self.config.transforms.resize
        self.transform_resize = get_resize_transform(self.img_sz)

        # params for augmentations during training
        with_aug = self.config.transforms.with_augmentation
        self.p_flip = self.config.transforms.p_flip if with_aug else 0
        self.p_crop = self.config.transforms.p_crop if with_aug else 0
        self.p_rot = self.config.transforms.p_rot if with_aug else 0

    def __len__(self):
        return self.total_frames

    def _plot_tactile_clip(self, clip):
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, self.num_frames, figsize=(20, 5))
        for i in range(self.num_frames):
            axs[i].imshow(clip[i].permute(1, 2, 0))
            axs[i].axis("off")
        plt.savefig("tactile_clip.png")
        plt.close()

    def __getitem__(self, idx):
        """Returns a single training item from the dataset as a dictionary."""
        idx += self.frames_concat_idx[-1]
        inputs = {}

        do_flip = random.random() < self.p_flip
        do_crop = random.random() < self.p_crop
        do_rot = random.random() < self.p_rot

        try:
            # get images
            image = self._get_tactile_images(idx, do_flip, do_crop, do_rot)
            inputs["image"] = image
            return inputs

        except Exception as ex:
            log.warning(f"Error in loading the image: {ex}, trying again")
            self.__getitem__(np.random.randint(0, self.__len__()))

    def _get_tactile_images(self, ref_frame, do_flip, do_crop, do_rot):
        if do_crop:
            crop_random_size = random.uniform(0.6, 0.9)
            crop_size = (int(self.img_sz[0]*crop_random_size), int(self.img_sz[1]*crop_random_size))

        if do_rot:
            max_angle = 15
            random_angle = random.random() * (max_angle*2) - max_angle
            mask = torch.ones(
                (1, self.img_sz[0], self.img_sz[1])
            )
            mask = TF.rotate(
                mask, max_angle, interpolation=transforms.InterpolationMode.BILINEAR
            )
            left = torch.argmax(mask[:, 0, :]).item()
            top = torch.argmax(mask[:, :, 0]).item()
            rot_coin = min(left, top)
            rot_size = (self.img_sz[0] - rot_coin, self.img_sz[1] - rot_coin)

        sample_images = []
        for i in self.frames_concat_idx:
            frame_index = np.clip(ref_frame - i, 0, self.total_frames - 1)

            image = self.loader(self.frames[frame_index], self.bg)
            image = self.transform_resize(image)

            if do_flip:
                image = TF.hflip(image)

            if do_crop:
                image = TF.center_crop(image, crop_size)
                image = transforms.Resize(self.img_sz, antialias=True)(image)

            if do_rot:
                image = TF.rotate(
                    image,
                    random_angle,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                )
                
                image = TF.center_crop(image, rot_size)
                image = transforms.Resize(self.img_sz, antialias=True)(image)

            sample_images.append(image)

        if DEBUG:
            self._plot_tactile_clip(sample_images)

        if self.out_format == "single_image":
            output = sample_images[0]
        elif self.out_format == "video":
            output = torch.stack(sample_images, dim=0)
        elif self.out_format == "concat_ch_img":
            output = torch.cat(sample_images, dim=0)
        return output
