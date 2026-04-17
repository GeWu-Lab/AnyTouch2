# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

from omegaconf import DictConfig
import numpy as np
import pickle
import cv2

import torch
from torch.utils import data
import torchvision.transforms.functional as TF
from torchvision import transforms

from sparsh.tactile_ssl.data.digit.utils import (
    get_path_images,
    pil_loader,
    get_resize_transform,
    get_path_dataset,
)

LABELS_SLIP = {0: "no_contact", 1: "no_shear", 2: "shear", 3: "partial_slip", 4: "slip"}


class DigitSlipDataset(data.Dataset):
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

        self.with_markers = self.config.with_markers  #
        if self.with_markers:
            self.remove_bg = False
            self.frames_concat_idx = [0]

        # transforms
        self.img_sz = self.config.transforms.resize
        self.transform_resize = get_resize_transform(self.img_sz)

        # background
        self.bg = cv2.imread(self.path_images[0]) if self.remove_bg else None

        # force labels
        path_dataset = get_path_dataset(self.config, dataset_name)

        with open(path_dataset + "/labels_slip.pkl", "rb") as f:
            self.gt_slip = pickle.load(f)

    def __len__(self):
        return len(self.path_images) - (self.d_frames * 2)

    def __getitem__(self, idx):
        idx += self.d_frames
        inputs = {}

        try:
            inputs["image"] = self._get_color(idx)
            inputs["label"] = self.gt_slip[idx]
            inputs["category_label"] = LABELS_SLIP[self.gt_slip[idx]]
            return inputs

        except Exception as e:
            self.__getitem__(np.random.randint(0, self.__len__()))

    def _get_color(self, ref_frame):
        sample_images = []
        for i in self.frames_concat_idx:
            frame_index = ref_frame + i
            image = self.loader(self.path_images[frame_index], self.bg)
            image = self.transform_resize(image)
            sample_images.append(image)

        if self.with_markers:
            image = self.loader(self.path_images[0], self.bg)
            ref_frame = self.transform_resize(image)
            sample_images.append(ref_frame)

        image = torch.cat(sample_images, dim=0)
        return image
