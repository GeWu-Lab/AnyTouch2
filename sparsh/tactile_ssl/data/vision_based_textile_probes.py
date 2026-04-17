# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import cv2
from omegaconf import DictConfig
import numpy as np

import torch
from torch.utils import data
import torchvision.transforms.functional as TF
from torchvision import transforms
from scipy.spatial.transform import Rotation as R

from sparsh.tactile_ssl.data.digit.utils import (
    load_textile_dataset,
    load_sample_from_buf,
    get_resize_transform,
)

DEBUG = False

class TextileDataset(data.Dataset):
    def __init__(self, config: DictConfig, dataset_name: str):

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
        self.dataset, metadata = load_textile_dataset(self.config, dataset_name)
        self.n_samples = len(self.dataset)
        self.loader = load_sample_from_buf
        self.label = int(metadata.split("\n")[0].split("label:")[-1])
        self.class_name = metadata.split("\n")[1].split("class_name: ")[-1]

        # transforms
        self.img_sz = self.config.transforms.resize
        self.transform_resize = get_resize_transform(self.img_sz)

    def __len__(self):
        return self.n_samples
    
    def _plot_tactile_clip(self, clip):
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, self.num_frames, figsize=(20, 5))
        for i in range(self.num_frames):
            axs[i].imshow(clip[i].permute(1, 2, 0))
            axs[i].axis("off")
        plt.savefig("tactile_clip.png")
    
    def __getitem__(self, idx: int):
        try:
            inputs = {}
            inputs["image"] = self._get_tactile_images(idx)
            inputs["textile_label"] = self.label
            return inputs
        except Exception as e:
            self.__getitem__(np.random.randint(self.n_samples))
    
    def _get_tactile_images(self, idx):
        sample_images = []
        for i in self.frames_concat_idx:
            idx_sample = np.clip(idx - i, 0, self.n_samples - 1)
            image = self.loader(self.dataset[idx_sample])
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