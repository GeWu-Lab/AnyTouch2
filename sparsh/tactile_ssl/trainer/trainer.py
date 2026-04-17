# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections.abc import Mapping
from functools import partial
from typing import Any, Iterable, List, Literal, Optional, Tuple, Union, cast

import lightning as L
import numpy as np
import torch
from lightning.fabric.accelerators import Accelerator
from lightning.fabric.strategies import Strategy
from lightning_utilities import apply_to_collection
from tqdm import tqdm

from sparsh.tactile_ssl.algorithm.module import Module  # noqa F401
from sparsh.tactile_ssl.utils.logging import get_pylogger
from sparsh.tactile_ssl.utils.signal_connector import SignalConnector

log = get_pylogger(__name__)


class Trainer:
    def __init__(
        self,
        accelerator: Union[str, Accelerator] = "auto",
        strategy: Union[str, Strategy] = "auto",
        devices: Union[List[int], str, int] = "auto",
        precision: Union[str, int] = "32-true",
        plugins: Optional[Union[str, Any]] = None,
        callbacks: Optional[Union[List[Any], Any]] = None,
        wandb_logger=None,
        max_epochs: Optional[int] = 1000,
        max_steps: Optional[int] = None,
        grad_accum_steps: int = 1,
        grad_clip_norm: Optional[float] = 10.0,
        limit_train_batches: Union[int, float] = float("inf"),
        limit_val_batches: Union[int, float] = float("inf"),
        validation_frequency: int = 1,
        sanity_validate: bool = False,
        use_distributed_sampler: bool = True,
        save_checkpoint_dir: str = "./checkpoints",
        checkpoint_frequency: int = 1,
        log_frequency: int = 1,
        checkpoint_interval_type: Literal['linear', 'log'] = 'linear',
        max_task_checkpoints: Optional[int] = None,
        save_probe_weights_only: Optional[bool] = False,
    ) -> None:
        """
        Args:
            accelerator: The hardware to run on. Possible choices are:
                ``"cpu"``, ``"cuda"``, ``"mps"``, ``"gpu"``, ``"tpu"``, ``"auto"``.
            strategy: Strategy for how to run across multiple devices. Possible choices are:
                ``"dp"``, ``"ddp"``, ``"ddp_spawn"``, ``"deepspeed"``, ``"fsdp"``.
            devices: Number of devices to train on (``int``),
                which GPUs to train on (``list`` or ``str``), or ``"auto"``.
                The value applies per node.
            precision: Double precision (``"64"``), full precision (``"32"``), half precision AMP (``"16-mixed"``),
                or bfloat16 precision AMP (``"bf16-mixed"``).
            plugins: One or several custom plugins
            callbacks: A single callback or a list of callbacks. The following hooks are supported:
                - on_train_epoch_start
                - on train_epoch_end
                - on_train_batch_start
                - on_train_batch_end
                - on_before_backward
                - on_after_backward
                - on_before_zero_grad
                - on_before_optimizer_step
                - on_validation_model_eval
                - on_validation_model_train
                - on_validation_epoch_start
                - on_validation_epoch_end
                - on_validation_batch_start
                - on_validation_batch_end

            loggers: A single logger or a list of loggers. See :meth:`~lightning.fabric.fabric.Fabric.log` for more
                information.

            max_epochs: The maximum number of epochs to train
            max_steps: The maximum number of (optimizer) steps to train
            grad_accum_steps: How many batches to process before each optimizer step
            limit_train_batches: Limits the number of train batches per epoch
                If greater than number of batches in the dataloader, this has no effect.
            limit_val_batches: Limits the number of validation batches per epoch.
                If greater than number of batches in the dataloader, this has no effect.
            validation_frequency: How many epochs to run before each validation epoch.
            use_distributed_sampler: Wraps the sampler of each dataloader with a respective distributed-aware sampler
                in case of distributed training.
            checkpoint_dir: Directory to store checkpoints to.
            checkpoint_frequency: How many epochs to run before each checkpoint is written.

        Warning:
            callbacks written for the lightning trainer (especially making assumptions on the trainer), won't work!

        """

        self.fabric = L.Fabric(
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            precision=precision,
            plugins=plugins,
            callbacks=callbacks,
        )
        self._connector = self.fabric._connector
        self._signal_connector = SignalConnector(self)
        self._signal_connector.register_signal_handlers()
        self.wandb = wandb_logger

        self.global_step = 0
        self.global_val_step = 0
        self.grad_accum_steps: int = grad_accum_steps
        self.grad_clip_norm = grad_clip_norm
        self.current_epoch = 0
        self.stage: Optional[Literal['train', 'val']] = None

        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.should_stop = False
        self.sanity_validate = sanity_validate

        self.state = None
        self.training_state = None

        # ensures limit_X_batches is either int or inf
        if not isinstance(limit_train_batches, int):
            assert limit_train_batches == float("inf")

        if not isinstance(limit_val_batches, int):
            assert limit_val_batches == float("inf")

        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.validation_frequency = validation_frequency
        self.use_distributed_sampler = use_distributed_sampler
        self._current_train_return: Union[torch.Tensor, Mapping[str, Any]] = {}
        self._current_val_return: Optional[Union[torch.Tensor, Mapping[str, Any]]] = {}

        self.checkpoint_dir = save_checkpoint_dir
        self.checkpoint_frequency = checkpoint_frequency
        self.log_frequency = log_frequency
        self.save_probe_weights_only = save_probe_weights_only
        self.max_task_checkpoints = max_task_checkpoints

        if max_task_checkpoints is not None:
            if checkpoint_interval_type == 'log':
                self.task_ep_save_ckpt =  np.geomspace(1, self.max_epochs, max_task_checkpoints, dtype=np.int32)
                self.task_ep_save_ckpt[0] = 0
            else:
                self.task_ep_save_ckpt = np.linspace(0, self.max_epochs, max_task_checkpoints, dtype=np.int)


    def fit(
        self,
        module: Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        ckpt_path: Optional[str] = None,
    ):
        """The main entrypoint of the trainer, triggering the actual training.

        Args:
            module: the LightningModule to train.
                Can have the same hooks as :attr:`callbacks` (see :meth:`MyCustomTrainer.__init__`).
            train_loader: the training dataloader. Has to be an iterable returning batches.
            val_loader: the validation dataloader. Has to be an iterable returning batches.
                If not specified, no validation will run.
            ckpt_path: Path to previous checkpoints to resume training from.
                If specified, will always look for the latest checkpoint within the given directory.

        """
        self.fabric.launch()

        # setup dataloaders
        train_loader = self.fabric.setup_dataloaders(
            train_loader, use_distributed_sampler=self.use_distributed_sampler
        )
        if val_loader is not None:
            val_loader = self.fabric.setup_dataloaders(
                val_loader, use_distributed_sampler=self.use_distributed_sampler
            )
        # setup module and optimizer
        if isinstance(self.fabric.strategy, L.fabric.strategies.fsdp.FSDPStrategy):
            # currently, there is no way to support fsdp with module.configure_optimizers in fabric
            # as it would require fabric to hold a reference to the module, which we don't want to.
            raise NotImplementedError("FSDP not supported at the moment")

        module = self.fabric.setup(module)

        optimizer, scheduler_cfg, wd_scheduler_cfg = self._parse_optimizers_schedulers(
            module.configure_optimizers(
                num_iterations_per_epoch=len(train_loader) // self.grad_accum_steps,
                num_epochs=self.max_epochs,
            )
        )

        assert (
            optimizer is not None
        ), "Could not parse optimizer from SSL module: {module.__class__.__name__}"
        optimizer = self.fabric.setup_optimizers(optimizer)

        # assemble state (current epoch and global step will be added in save)
        state = {"model": module, "optim": optimizer, "scheduler": scheduler_cfg}

        if wd_scheduler_cfg is not None:
            state.update(wd_scheduler=wd_scheduler_cfg)

        # load last checkpoint if available
        if ckpt_path is not None and os.path.isdir(ckpt_path):
            latest_checkpoint_path = self.get_latest_checkpoint(ckpt_path)
            if latest_checkpoint_path is not None:
                log.info(f"Loading latest checkpoint to resume training: {latest_checkpoint_path}")
                self.load(state, latest_checkpoint_path)

                # check if we even need to train here
                if (
                    self.max_epochs is not None
                    and self.current_epoch >= self.max_epochs
                ):
                    self.should_stop = True

        self.state = state

        # Always start with one validation loop first
        # if self.sanity_validate and self.should_validate:
        #     self.val_loop(module, val_loader, limit_batches=self.limit_val_batches)

        while not self.should_stop:
            self.train_loop(
                module,
                optimizer,
                train_loader,
                limit_batches=self.limit_train_batches,
                scheduler_cfg=scheduler_cfg,
                wd_scheduler_cfg=wd_scheduler_cfg,
            )

            if self.should_validate:
                self.val_loop(module, val_loader, limit_batches=self.limit_val_batches)

            self.step_scheduler(
                scheduler_cfg, level="epoch", current_value=self.current_epoch
            )
            self.step_wd_scheduler(
                wd_scheduler_cfg, level="epoch", current_value=self.current_epoch
            )

            self.current_epoch += 1

            # stopping condition on epoch level
            if self.max_epochs is not None and self.current_epoch >= self.max_epochs:
                self.should_stop = True

            self.save_latest_checkpoint()

            if self.should_save:
                self.save_checkpoint()

        # reset for next fit call
        self.should_stop = False

    def train_loop(
        self,
        module: Module,
        optimizer: torch.optim.Optimizer,
        train_loader: torch.utils.data.DataLoader,
        limit_batches: Union[int, float] = float("inf"),
        scheduler_cfg: Optional[
            Mapping[str, Union[L.fabric.utilities.types.LRScheduler, bool, str, int]]
        ] = None,
        wd_scheduler_cfg: Optional[Mapping[str, Union[object, bool, str, int]]] = None,
    ):
        """The training loop running a single training epoch.

        Args:
            module: the LightningModule to train
            optimizer: the optimizer, optimizing the LightningModule.
            train_loader: The dataloader yielding the training batches.
            limit_batches: Limits the batches during this training epoch.
                If greater than the number of batches in the ``train_loader``, this has no effect.
            scheduler_cfg: The learning rate scheduler configuration.
                Have a look at :meth:`~lightning.pytorch.core.LightningModule.configure_optimizers`
                for supported values.

        """
        self.training_state = 'train'
        module.on_train_epoch_start()
        self.fabric.call("on_train_epoch_start")

        # print('scheduler_cfg')
        # print(scheduler_cfg)
        # exit(0)
        # print(optimizer)

        iterable = self.progbar_wrapper(
            train_loader,
            total=min(len(train_loader), limit_batches),
            desc=f"Epoch {self.current_epoch}",
        )

        for batch_idx, batch in enumerate(iterable):
            # end epoch if stopping training completely or max batches for this epoch reached
            if self.should_stop or batch_idx >= limit_batches:
                break

            module.on_train_batch_start(batch, batch_idx)
            self.fabric.call("on_train_batch_start", batch, batch_idx)

            # check if optimizer should step in gradient accumulation
            should_optim_step = self.global_step % self.grad_accum_steps == 0
            if should_optim_step:
                # currently only supports a single optimizer
                self.fabric.call("on_before_optimizer_step", optimizer, 0)


                # optimizer step runs train step internally through closure
                self.training_step(module=module, batch=batch, batch_idx=batch_idx)
                if self.grad_clip_norm:
                    self.fabric.clip_gradients(
                        module, optimizer, max_norm=self.grad_clip_norm
                    )
                optimizer.step()
                optimizer.zero_grad()
                self.fabric.call("on_before_zero_grad", optimizer)

            else:
                # gradient accumulation -> no optimizer step
                self.training_step(module=module, batch=batch, batch_idx=batch_idx)

            module.on_train_batch_end(
                self._current_train_return, batch, batch_idx, self
            )
            self.fabric.call(
                "on_train_batch_end", self._current_train_return, batch, batch_idx
            )

            # this guard ensures, we only step the scheduler once per global step
            if should_optim_step:
                self.step_scheduler(
                    scheduler_cfg, level="step", current_value=self.global_step
                )
                self.step_wd_scheduler(
                    wd_scheduler_cfg, level="step", current_value=self.global_step
                )
            # add output values to progress bar
            self._format_iterable(iterable, self._current_train_return["loss"], "train")

            # only increase global step if optimizer stepped
            self.global_step += int(should_optim_step)

            # stopping criterion on step level
            if self.max_steps is not None and self.global_step >= self.max_steps:
                self.should_stop = True
                break

        module.on_train_epoch_end(self)
        self.fabric.call("on_train_epoch_end")

    def val_loop(
        self,
        module: Module,
        val_loader: Optional[torch.utils.data.DataLoader],
        limit_batches: Union[int, float] = float("inf"),
    ):
        """The validation loop ruunning a single validation epoch.

        Args:
            module: the LightningModule to evaluate
            val_loader: The dataloader yielding the validation batches.
            limit_batches: Limits the batches during this validation epoch.
                If greater than the number of batches in the ``val_loader``, this has no effect.

        """
        self.training_state = 'val'
        # no validation if val_loader wasn't passed
        if val_loader is None:
            return

        # no validation but warning if val_loader was passed, but validation_step not implemented
        if val_loader is not None and not hasattr(module, "validation_step"):
            L.fabric.utilities.rank_zero_warn(
                "Your LightningModule does not have a validation_step implemented, "
                "but you passed a validation dataloder. Skipping Validation."
            )
            return
        self.fabric.call("on_validation_module_eval")  # calls `module.eval()`

        torch.set_grad_enabled(False)

        self.fabric.call("on_validation_epoch_start")

        iterable = self.progbar_wrapper(
            val_loader, total=min(len(val_loader), limit_batches), desc="Validation"
        )

        for batch_idx, batch in enumerate(iterable):
            # end epoch if stopping training completely or max batches for this epoch reached
            if self.should_stop or batch_idx >= limit_batches:
                break
            self.fabric.call("on_validation_batch_start", batch, batch_idx)

            out = module.validation_step(batch, batch_idx)
            # avoid gradients in stored/accumulated values -> prevents potential OOM
            out = apply_to_collection(out, torch.Tensor, lambda x: x.detach())

            module.on_validation_batch_end(out, batch, batch_idx, self)
            self.fabric.call("on_validation_batch_end", out, batch, batch_idx)
            self._current_val_return = out

            self._format_iterable(iterable, self._current_val_return["loss"], "val")

            self.global_val_step += 1

        module.on_validation_epoch_end(self)
        self.fabric.call("on_validation_epoch_end")

        self.fabric.call("on_validation_model_train")
        torch.set_grad_enabled(True)

    def training_step(self, module: Module, batch: Any, batch_idx: int) -> torch.Tensor:
        """A single training step, running forward and backward. The optimizer step is called separately, as this is
        given as a closure to the optimizer step.

        Args:
            model: the lightning module to train
            batch: the batch to run the forward on
            batch_idx: index of the current batch w.r.t the current epoch

        """
        outputs: Union[torch.Tensor, Mapping[str, Any]] = module.training_step(
            batch, batch_idx=batch_idx
        )

        loss = outputs if isinstance(outputs, torch.Tensor) else outputs["loss"]

        self.fabric.call("on_before_backward", loss)
        self.fabric.backward(loss)
        self.fabric.call("on_after_backward")

        # avoid gradients in stored/accumulated values -> prevents potential OOM
        self._current_train_return = apply_to_collection(
            outputs, dtype=torch.Tensor, function=lambda x: x.detach()
        )

        return loss

    def step_wd_scheduler(
        self,
        wd_scheduler_cfg: Optional[object],
        level: Literal["step", "epoch"],
        current_value: int,
    ) -> None:
        if wd_scheduler_cfg is None:
            return

        if wd_scheduler_cfg["interval"] != level:
            return

        if current_value % cast(int, wd_scheduler_cfg["frequency"]) != 0:
            return

        new_wd = wd_scheduler_cfg["wd_scheduler"].step()
        self.wandb.log({"weight_decay": new_wd})

    def step_scheduler(
        self,
        scheduler_cfg: Optional[
            Mapping[str, Union[L.fabric.utilities.types.LRScheduler, bool, str, int]]
        ],
        level: Literal["step", "epoch"],
        current_value: int,
    ) -> None:
        """Steps the learning rate scheduler if necessary.

        Args:
            scheduler_cfg: The learning rate scheduler configuration.
                Have a look at :meth:`lightning.pytorch.LightningModule.configure_optimizers` for supported values.
            level: whether we are trying to step on epoch- or step-level
            current_value: Holds the current_epoch if ``level==epoch``, else holds the ``global_step``

        """

        # no scheduler
        if scheduler_cfg is None:
            return

        # wrong interval (step vs. epoch)
        if scheduler_cfg["interval"] != level:
            return

        # right interval, but wrong step wrt frequency
        if current_value % cast(int, scheduler_cfg["frequency"]) != 0:
            return

        # assemble potential monitored values
        possible_monitor_vals = {None: None}
        if isinstance(self._current_train_return, torch.Tensor):
            possible_monitor_vals.update("train_loss", self._current_train_return)
        elif isinstance(self._current_train_return, Mapping):
            possible_monitor_vals.update(
                {"train_" + k: v for k, v in self._current_train_return.items()}
            )

        if isinstance(self._current_val_return, torch.Tensor):
            possible_monitor_vals.update("val_loss", self._current_val_return)
        elif isinstance(self._current_val_return, Mapping):
            possible_monitor_vals.update(
                {"val_" + k: v for k, v in self._current_val_return.items()}
            )

        try:
            monitor = possible_monitor_vals[
                cast(Optional[str], scheduler_cfg["monitor"])
            ]
        except KeyError as ex:
            possible_keys = list(possible_monitor_vals.keys())
            raise KeyError(
                f"monitor {scheduler_cfg['monitor']} is invalid. Possible values are {possible_keys}."
            ) from ex

        # rely on model hook for actual step
        if monitor is None:
            scheduler_cfg["scheduler"].step()
        else:
            scheduler_cfg["scheduler"].step(monitor)

        for i, _ in enumerate(scheduler_cfg["scheduler"].optimizer.param_groups):
            self.wandb.log({f"lr_{i}": scheduler_cfg["scheduler"].get_last_lr()[i]})

    @property
    def should_validate(self) -> bool:
        """Whether to currently run validation."""
        return self.current_epoch % self.validation_frequency == 0

    @property
    def should_save(self) -> bool:
        """Whether to currently save a checkpoint."""
        if self.max_task_checkpoints is not None: 
            return self.current_epoch in self.task_ep_save_ckpt 
        else:    
            return self.current_epoch % self.checkpoint_frequency == 0
        
    
    @property
    def should_log(self) -> bool: 
        if self.training_state == 'val': 
            return self.global_val_step % self.log_frequency == 0 
        return self.global_step % self.log_frequency == 0

    def progbar_wrapper(self, iterable: Iterable, total: int, **kwargs: Any):
        """Wraps the iterable with tqdm for global rank zero.

        Args:
            iterable: the iterable to wrap with tqdm
            total: the total length of the iterable, necessary in case the number of batches was limited.

        """
        if self.fabric.is_global_zero:
            return tqdm(iterable, total=total, **kwargs)
        return iterable

    def load(self, state: Optional[Mapping], path: str) -> None:
        """Loads a checkpoint from a given file into state.

        Args:
            state: a mapping contaning model, optimizer and lr scheduler
            path: the path to load the checkpoint from

        """
        if state is None:
            state = {}

        remainder = self.fabric.load(path, state)
        self.global_step = remainder.pop("global_step")
        self.current_epoch = remainder.pop("current_epoch")

        if remainder:
            raise RuntimeError(f"Unused Checkpoint Values: {remainder}")
        log.info(f"Loaded checkpoint from {path}")

    def save_checkpoint(self, state: Optional[Mapping] = None) -> None:
        """Saves a checkpoint to the ``checkpoint_dir``

        Args:
            state: A mapping containing model, optimizer and lr scheduler.

        """

        if self.max_task_checkpoints is not None: 
            if self.save_probe_weights_only:
                # save only model weights that start with model_task
                state_dict = self.state["model"].state_dict()
                task_keys = [k for k in state_dict.keys() if k.startswith("model_task")]
                state_dict = {k: state_dict[k] for k in task_keys}
                torch.save(state_dict, os.path.join(self.checkpoint_dir, f"epoch-{self.current_epoch:04d}.pth"))
            else:
                state_dict = self.state["model"].state_dict()
                torch.save(state_dict, os.path.join(self.checkpoint_dir, f"epoch-{self.current_epoch:04d}.pth"))

        else:
            self.fabric.save(
                os.path.join(self.checkpoint_dir, f"epoch-{self.current_epoch:04d}.ckpt"),
                self.state,
            )
    
    def save_latest_checkpoint(self, state: Optional[Mapping] = None) -> None:
        """Saves a checkpoint to the ``checkpoint_dir``

        Args:
            state: A mapping containing model, optimizer and lr scheduler.

        """
        self.state.update(
            global_step=self.global_step, current_epoch=self.current_epoch
        )

        self.fabric.save(
            os.path.join(self.checkpoint_dir, "last.ckpt"), self.state,
        )

    @staticmethod
    def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
        """Returns the latest checkpoint from the ``checkpoint_dir``

        Args:
            checkpoint_dir: the directory to search for checkpoints

        """
        if not os.path.isdir(checkpoint_dir):
            log.warning(f"Checkpoint should be a directory containing the checkpoints to resume training")
            return None

        items = sorted(os.listdir(checkpoint_dir))
        last_ckpt = "last.ckpt"

        if last_ckpt not in items:
            return None

        return os.path.join(checkpoint_dir, last_ckpt)

    def _parse_optimizers_schedulers(
        self, configure_optim_output
    ) -> Tuple[
        Optional[L.fabric.utilities.types.Optimizable],
        Optional[
            Mapping[str, Union[L.fabric.utilities.types.LRScheduler, bool, str, int]]
        ],
        Optional[Mapping[str, Union[object, bool, str, int]]],
    ]:
        """Recursively parses the output of :meth:`lightning.pytorch.LightningModule.configure_optimizers`.

        Args:
            configure_optim_output: The output of ``configure_optimizers``.
                For supported values, please refer to :meth:`lightning.pytorch.LightningModule.configure_optimizers`.

        """
        _lr_sched_defaults = {
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss",
        }

        # single optimizer
        if isinstance(configure_optim_output, L.fabric.utilities.types.Optimizable):
            return configure_optim_output, None, None

        # single lr scheduler
        if isinstance(configure_optim_output, L.fabric.utilities.types.LRScheduler):
            return (
                None,
                _lr_sched_defaults.update(scheduler=configure_optim_output),
                None,
            )

        # single lr scheduler config
        if isinstance(configure_optim_output, Mapping):
            _lr_sched_defaults.update(configure_optim_output)
            if "scheduler" in _lr_sched_defaults:
                return None, _lr_sched_defaults, None
            elif "wd_scheduler" in _lr_sched_defaults:
                return None, None, _lr_sched_defaults
            else:
                raise ValueError(
                    "Invalid configuration for lr scheduler or weight decay scheduler"
                )

        # list or tuple
        if isinstance(configure_optim_output, (list, tuple)):
            if all(
                isinstance(_opt_cand, L.fabric.utilities.types.Optimizable)
                for _opt_cand in configure_optim_output
            ):
                # single optimizer in list
                if len(configure_optim_output) == 1:
                    return configure_optim_output[0][0], None, None

                raise NotImplementedError(
                    "Trainer only supports a single optimizer for now"
                )

            if all(
                isinstance(_lr_cand, (L.fabric.utilities.types.LRScheduler, Mapping))
                for _lr_cand in configure_optim_output
            ):
                # single scheduler in list
                if len(configure_optim_output) == 1:
                    return (
                        None,
                        self._parse_optimizers_schedulers(configure_optim_output[0])[1],
                        None,
                    )

            # optimizer and lr scheduler
            elif len(configure_optim_output) == 2:
                opt_cands, lr_cands = (
                    self._parse_optimizers_schedulers(configure_optim_output[0])[0],
                    self._parse_optimizers_schedulers(configure_optim_output[1])[1],
                )
                return opt_cands, lr_cands, None
            else:
                assert (
                    len(configure_optim_output) == 3
                ), "Invalid optimizer configuration"
                opt_cands, lr_cands, wd_cands = (
                    self._parse_optimizers_schedulers(configure_optim_output[0])[0],
                    self._parse_optimizers_schedulers(configure_optim_output[1])[1],
                    self._parse_optimizers_schedulers(configure_optim_output[2])[2],
                )
                return opt_cands, lr_cands, wd_cands

        return None, None, None

    @staticmethod
    def _format_iterable(
        prog_bar,
        candidates: Optional[
            Union[torch.Tensor, Mapping[str, Union[torch.Tensor, float, int]]]
        ],
        prefix: str,
    ):
        """Adds values as postfix string to progressbar.

        Args:
            prog_bar: a progressbar (on global rank zero) or an iterable (every other rank).
            candidates: the values to add as postfix strings to the progressbar.
            prefix: the prefix to add to each of these values.

        """
        if isinstance(prog_bar, tqdm) and candidates is not None:
            postfix_str = ""
            float_candidates = apply_to_collection(
                candidates, torch.Tensor, lambda x: x.item()
            )
            if isinstance(candidates, torch.Tensor):
                postfix_str += f" {prefix}_loss: {float_candidates:.3f}"
            elif isinstance(candidates, Mapping):
                for k, v in float_candidates.items():
                    postfix_str += f" {prefix}_{k}: {v:.3f}"

            if postfix_str:
                prog_bar.set_postfix_str(postfix_str)

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        dataset = self.train_dataloader()
        if self.trainer.max_steps:
            return self.trainer.max_steps

        dataset_size = (
            self.trainer.limit_train_batches
            if self.trainer.limit_train_batches != 0
            else len(dataset)
        )

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_batch_size = (
            dataset.batch_size * self.trainer.accumulate_grad_batches * num_devices
        )
        return (dataset_size // effective_batch_size) * self.trainer.max_epochs
