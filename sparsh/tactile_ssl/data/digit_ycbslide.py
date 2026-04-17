# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

from omegaconf import DictConfig
import numpy as np
import random

import torch
from torch.utils import data
import torchvision.transforms.functional as TF
from torchvision import transforms

from sparsh.tactile_ssl.data.digit.utils import (
    get_path_images,
    pil_loader,
    get_resize_transform,
    get_bg_img,
    get_digit_intrinsics,
)

from sparsh.tactile_ssl.utils.logging import get_pylogger

log = get_pylogger()


class DigitYCBSlideDataset(data.Dataset):
    def __init__(
        self,
        config: DictConfig,
        dataset_name: str,
    ):
        super().__init__()
        self.config = config
        self.dataset_name = dataset_name

        self.remove_bg = self.config.remove_bg  # if remove bg
        self.d_frames = self.config.d_frames  # temporal distance
        self.frames_concat_idx = [
            0,
            -self.d_frames,
        ]  # frames to concatenate, relative to the current index

        # path images
        self.path_images = get_path_images(self.config, dataset_name)
        self.loader = pil_loader

        # transforms
        self.img_sz = self.config.transforms.resize
        self.transform_resize = get_resize_transform(self.img_sz)

        # params for augmentations during training
        with_aug = self.config.transforms.with_augmentation
        self.p_flip = self.config.transforms.p_flip if with_aug else 0
        self.p_crop = self.config.transforms.p_crop if with_aug else 0
        self.p_rot = self.config.transforms.p_rot if with_aug else 0

        # background
        self.bg = get_bg_img(self.config, dataset_name, self.remove_bg)

        # intrinsics
        self.K = get_digit_intrinsics(self.img_sz)
        self.inv_K = np.linalg.pinv(self.K)

    def __len__(self):
        return len(self.path_images) - (self.d_frames * 2)

    def __getitem__(self, idx):
        """Returns a single training item from the dataset as a dictionary."""
        idx += self.d_frames
        inputs = {}

        do_flip = random.random() < self.p_flip
        do_crop = random.random() < self.p_crop
        do_rot = random.random() < self.p_rot

        try:
            # get images
            image = self._get_digit_images(idx, do_flip, do_crop, do_rot)
            inputs["image"] = image
            return inputs

        except Exception as ex:
            log.warning(f"Error in loading the image: {ex}, trying again")
            self.__getitem__(np.random.randint(0, self.__len__()))

    def _get_digit_images(self, ref_frame, do_flip, do_crop, do_rot):
        if do_crop:
            crop_random_size = int(random.uniform(0.6, 0.9) * self.img_sz[0])
            max_size = self.img_sz[0] - crop_random_size
            crop_left = int(random.random() * max_size)
            crop_top = int(random.random() * max_size)

        if do_rot:
            random_angle = random.random() * 20 - 10
            mask = torch.ones(
                (1, self.img_sz[0], self.img_sz[1])
            )  # useful for the resize at the end
            mask = TF.rotate(
                mask, random_angle, interpolation=transforms.InterpolationMode.BILINEAR
            )
            left = torch.argmax(mask[:, 0, :]).item()
            top = torch.argmax(mask[:, :, 0]).item()
            rot_coin = min(left, top)
            rot_size = self.img_sz[0] - 2 * rot_coin

        sample_images = []
        for i in self.frames_concat_idx:
            frame_index = ref_frame + i

            image = self.loader(self.path_images[frame_index], self.bg)
            image = self.transform_resize(image)

            if do_flip:
                image = TF.hflip(image)

            if do_crop:
                image = TF.crop(
                    image, crop_top, crop_left, crop_random_size, crop_random_size
                )
                image = transforms.Resize(self.img_sz, antialias=True)(image)

            if do_rot:
                image = TF.rotate(
                    image,
                    random_angle,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                )
                image = TF.crop(image, rot_coin, rot_coin, rot_size, rot_size)
                image = transforms.Resize(self.img_sz, antialias=True)(image)

            sample_images.append(image)

        image = torch.cat(sample_images, dim=0)
        return image
