# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Optional
from tactile_ssl import algorithm
from torch.utils import data


class TestTaskSL:
    def __init__(
        self,
        device,
        module: algorithm.Module,
    ):
        self.device = device
        self.module = module
        self.module.to(device)
        self.module.requires_grad_(False)
        self.module.eval()

    def set_test_params(self, task, sensor, ckpt, dataset_name, path_outputs, config=None):
        self.config = config
        self.task = task.split(sensor)[1][1:]
        self.epoch = int(ckpt.split("-")[-1].split(".")[0])
        dataset_name = str(dataset_name)
        self.dataset_name = (
            dataset_name.replace("/", "_") if "/" in dataset_name else dataset_name
        )
        self.path_outputs = f"{path_outputs}/{task}/{self.dataset_name}/"
        self.path_output_model = f"{path_outputs}/{task}/"
        os.makedirs(self.path_outputs, exist_ok=True)

    def run_model(self, dataset: data.Dataset, dataloader: data.DataLoader):
        pass

    def get_overall_metrics(
        self, dataset: data.Dataset, over_all_outputs: bool = False
    ):
        pass

    def make_plots(self, dataset: data.Dataset, params: Optional[dict] = None):
        pass

    def make_video(self, dataset: data.Dataset, params: Optional[dict] = None):
        pass
