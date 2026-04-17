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
    load_dataset_poses,
    load_sample_from_buf,
    get_resize_transform,
)

DEBUG = False

class PoseDataset(data.Dataset):
    def __init__(self, config: DictConfig, dataset_name: str, is_train: bool = False):

        super().__init__()
        self.config = config
        self.dataset_name = dataset_name
        self.t_stride = config.rel_pose_t_window
        self.finger_type = config.finger_type
        self.is_anytouch = config.is_anytouch
        # print(config)
        self.is_t3 = config.is_t3 if 'is_t3' in config else False
        self.is_unitouch = config.is_unitouch if 'is_unitouch' in config else False
        print(f"Is AnyTouch: {self.is_anytouch}, Is T3: {self.is_t3}")
        self.sensor = config.sensor if hasattr(config, "sensor") else "digit"

        assert self.finger_type in ["index", "middle", "ring"], ValueError(
            "config.finger_type should be 'index', 'middle', 'ring' or None"
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
        self.num_frames = (
            1 if self.out_format == "single_image" else self.config.num_frames
        )
        self.frames_concat_idx = np.arange(
            0, self.num_frames * frame_stride, frame_stride
        )

        # load dataset
        self.dataset_digit, self.dataset_poses = load_dataset_poses(
            self.config, dataset_name, self.finger_type, self.t_stride
        )
        self.loader = load_sample_from_buf

        # discretize poses and generate class labels
        self.dataset_poses_labels, self.bins_translation, self.bins_rotation = (
            self.discretize_poses()
        )
        self.n_samples = len(self.dataset_poses)

        # transforms
        self.img_sz = self.config.transforms.resize
        self.transform_resize = get_resize_transform(self.img_sz, is_anytouch=self.is_anytouch, is_t3=self.is_t3, sensor=self.sensor, is_train=is_train, is_unitouch=self.is_unitouch)

        # background
        self.bg = None
        if self.remove_bg:
            self.bg = cv2.imread(
                f"{self.config.path_bgs_fingers}/digit_{self.finger_type}.png"
            )
        
        print(f"Dataset {self.dataset_name} loaded with {self.n_samples} samples.")

    def discretize_poses(self):
        t_xyz = self.dataset_poses[:, :3, 3]
        t_xy = t_xyz[:, [1, 0]]
        r_rpy = R.from_matrix(self.dataset_poses[:, :3, :3]).as_euler(
            "xyz", degrees=True
        )
        r_py = r_rpy[:, [1, 2]]

        # discretize translations X and Y
        ths_xy = np.array(self.config.bins_translation)
        ths_xy = np.concatenate([ths_xy[::-1] * -1, ths_xy])
        t_x_labels = np.ones_like(t_xy[:, 0], dtype=np.int64) * -1
        t_y_labels = np.ones_like(t_xy[:, 1], dtype=np.int64) * -1

        for i, th in enumerate(ths_xy):
            if i == 0:
                t_x_labels[t_xy[:, 0] < th] = i
                t_y_labels[t_xy[:, 1] < th] = i
            else:
                t_x_labels[(t_xy[:, 0] < th) & (t_xy[:, 0] >= ths_xy[i - 1])] = i
                t_y_labels[(t_xy[:, 1] < th) & (t_xy[:, 1] >= ths_xy[i - 1])] = i
        t_x_labels[t_xy[:, 0] >= ths_xy[-1]] = i + 1
        t_y_labels[t_xy[:, 1] >= ths_xy[-1]] = i + 1

        assert (
            np.where(t_x_labels == -1)[0].sum() == 0
            and np.where(t_y_labels == -1)[0].sum() == 0
        ), "There are unlabeled samples for translation X or Y"

        # discretize rotation Yaw
        ths_py = np.array(self.config.bins_rotation)
        ths_py = np.concatenate([ths_py[::-1] * -1, ths_py])
        r_y_labels = np.ones_like(r_py[:, 1], dtype=np.int64) * -1

        for i, th in enumerate(ths_py):
            if i == 0:

                r_y_labels[r_py[:, 1] < th] = i
            else:
                r_y_labels[(r_py[:, 1] < th) & (r_py[:, 1] >= ths_py[i - 1])] = i
        r_y_labels[r_py[:, 1] >= ths_py[-1]] = i + 1

        assert (
            np.where(r_y_labels == -1)[0].sum() == 0
        ), "There are unlabeled samples for rotation Yaw"

        dataset_poses_labels = {
            "t_x": t_x_labels,
            "t_y": t_y_labels,
            "r_y": r_y_labels,
        }
        return dataset_poses_labels, ths_xy, ths_py

    def __len__(self):
        return self.n_samples

    def _plot_tactile_clip(self, clip):
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, self.num_frames, figsize=(20, 5))
        for i in range(self.num_frames):
            if self.is_anytouch:
                temp_clip = clip[i].permute(1, 2, 0)
                image_mean = np.array([0.485, 0.456, 0.406])
                image_std = np.array([0.229, 0.224, 0.225])
                temp_clip = temp_clip * image_std + image_mean
                axs[i].imshow(temp_clip)
                axs[i].axis("off")
            else:
                axs[i].imshow(clip[i].permute(1, 2, 0))
                axs[i].axis("off")
        plt.savefig("tactile_clip.png")

    def __getitem__(self, idx):
        inputs = {}
        inputs["image"] = {f"digit_{self.finger_type}": self._get_tactile_images(idx)}
        inputs["pose_labels"] = {}
        inputs["pose_labels"]["tx"] = self.dataset_poses_labels["t_x"][idx]
        inputs["pose_labels"]["ty"] = self.dataset_poses_labels["t_y"][idx]
        inputs["pose_labels"]["yaw"] = self.dataset_poses_labels["r_y"][idx]
        return inputs

    def _get_tactile_images(self, idx: int):

        sample_images = []
        for i in self.frames_concat_idx:
            idx_sample = np.clip(idx - i, 0, self.n_samples - 1)
            image = self.loader(self.dataset_digit[idx_sample], self.bg, is_anytouch=(self.is_anytouch or self.is_t3 or self.is_unitouch), sensor=self.sensor, convert=self.is_anytouch)
            image = self.transform_resize(image)
            sample_images.append(image)

        # if self.out_format in ["video", "concat_ch_img"]:
        #     sample_images = sample_images[::-1]

        if DEBUG:
            self._plot_tactile_clip(sample_images)
            exit(0)

        if self.out_format == "single_image":
            output = sample_images[0]
        elif self.out_format == "video":
            output = torch.stack(sample_images, dim=0)
            if not self.is_anytouch and not self.is_t3 and not self.is_unitouch:
                output = output.permute(1, 0, 2, 3)
        elif self.out_format == "concat_ch_img":
            output = torch.cat(sample_images, dim=0)
        return output