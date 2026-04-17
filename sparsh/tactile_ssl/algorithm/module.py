# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Any, Tuple, Optional, Union
from abc import ABC, abstractmethod
import torch


class Module(ABC):
    @abstractmethod
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def configure_optimizers(
        self, num_iterations_per_epoch: int, num_epochs: int
    ) -> Tuple[
        torch.optim.Optimizer,
        Optional[Dict],
        Optional[Dict],
    ]:
        raise NotImplementedError

    def on_train_epoch_end(self, trainer_instance=None):
        pass

    def on_validation_epoch_end(self, trainer_instance=None):
        pass

    def on_train_batch_end(
        self, outputs: Dict, batch: Dict, batch_idx: int, trainer_instance=None
    ):
        pass

    def on_validation_batch_end(
        self, outputs: Dict, batch: Dict, batch_idx: int, trainer_instance=None
    ):
        pass

    def on_train_batch_start(self, batch: Dict, batch_idx: int):
        pass

    def on_train_epoch_start(self, trainer_instance=None):
        pass
