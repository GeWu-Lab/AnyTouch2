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

from sparsh.tactile_ssl.data.digit.utils import (
    load_feeling_success,
    load_sample_from_buf,
    get_resize_transform,
)


class GelsightGraspDataset(data.Dataset):
    def __init__(
        self,
        config: DictConfig,
        dataset_name: str,
    ):

        super().__init__()
        self.config = config
        self.dataset_name = dataset_name

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
        self.dataset = load_feeling_success(self.config, dataset_name)
        self.total_samples = len(self.dataset["is_gripping"])
        self.loader = load_sample_from_buf

        # transforms
        self.img_sz = self.config.transforms.resize
        self.transform_resize = get_resize_transform(self.img_sz)

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        try:
            label = int(self.dataset["is_gripping"][idx])
            inputs = {}
            inputs["image"] = self._get_tactile_images(idx)
            inputs["grasp_label"] = torch.tensor(label)
            return inputs
        except Exception as e:
            self.__getitem__(np.random.randint(0, self.__len__()))

    def _get_tactile_images(self, idx):
        sensor_id = "gelsightA" if torch.rand(1) >= 0.5 else "gelsightB"

        if self.out_format == "single_image":
            image = self.loader(self.dataset[f"{sensor_id}_during"][idx])
            image = self.transform_resize(image)
            return image

        elif self.out_format == "concat_ch_img":
            if torch.rand(1) >= 0.5:
                image1 = self.loader(self.dataset[f"{sensor_id}_during"][idx])
                image2 = self.loader(self.dataset[f"{sensor_id}_before"][idx])
            else:
                image1 = self.loader(self.dataset[f"{sensor_id}_after"][idx])
                image2 = self.loader(self.dataset[f"{sensor_id}_during"][idx])
            image1 = self.transform_resize(image1)
            image2 = self.transform_resize(image2)
            return torch.cat([image1, image2], dim=0)

        elif self.out_format == "video":
            if self.num_frames == 4:
                sample_images = []
                sample_images.append(
                    self.loader(self.dataset[f"{sensor_id}_after"][idx])
                )
                sample_images.append(
                    self.loader(self.dataset[f"{sensor_id}_during"][idx])
                )
                sample_images.append(
                    self.loader(self.dataset[f"{sensor_id}_before"][idx])
                )
                sample_images.append(
                    self.loader(self.dataset[f"{sensor_id}_before"][idx])
                )

                for i in range(4):
                    sample_images[i] = self.transform_resize(sample_images[i])
                return torch.stack(sample_images, dim=0)
            else:
                raise ValueError("Only 4 frames are supported for video output format.")
