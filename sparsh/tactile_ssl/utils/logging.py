# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

from typing import Sequence
import logging
from omegaconf import DictConfig, OmegaConf
import rich
from rich.tree import Tree
from rich.syntax import Syntax
from pathlib import Path
import numpy as np
import cv2

from lightning.fabric.utilities import rank_zero_only


def get_pylogger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    logging_levels = (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    )
    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


log = get_pylogger(__name__)


def print_config_tree(
    cfg: DictConfig,
    print_order: Sequence[str] = (
        "data",
        "model",
        "trainer",
        "paths",
    ),
    resolve: bool = False,
    save_to_file: bool = False,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
        print_order (Sequence[str], optional): Determines in what order config components are printed.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
        save_to_file (bool, optional): Whether to export config to the hydra output folder.
    """

    style = "dim"
    tree = Tree("CONFIG", style=style, guide_style=style)

    queue = []

    # add fields from `print_order` to queue
    for field in print_order:
        (
            queue.append(field)
            if field in cfg
            else log.warning(
                f"Field '{field}' not found in config. Skipping '{field}' config printing..."
            )
        )

    # add all the other fields to queue (not specified in `print_order`)
    for field in cfg:
        if field not in queue:
            queue.append(field)

    # generate config tree from queue
    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(Syntax(branch_content, "yaml"))

    # print config tree
    rich.print(tree)

    # save config tree to file
    if save_to_file:
        with open(Path(cfg.paths.output_dir, "config_tree.log"), "w") as file:
            rich.print(tree, file=file)


def img_logger(wandb, global_step, predictions, X=None, label="train"):
    nb_to_show = 5
    output_dim = (240, 320)

    if len(predictions.shape) == 4:
        predictions = predictions.unsqueeze(2)
        X = X.unsqueeze(2)

    # plot predictions
    B = predictions.shape[0]
    T = predictions.shape[2]
    idx = np.random.choice(range(B), nb_to_show, replace=False)

    tmp = predictions[idx].permute(0, 2, 3, 4, 1).cpu().numpy()
    preds = (tmp - tmp.min()) / (tmp.max() - tmp.min())
    preds = np.clip(preds, 0, 1).astype("float32")

    for t in range(T):
        wandb.log(
            {
                f"{label}/img_pred_t{t}": [
                    wandb.Image(
                        cv2.resize(im, output_dim),
                        caption="img_{}".format(i + 1),
                    )
                    for i, im in enumerate(preds[:, t])
                ],
                f"global_{label}_step": global_step,
            }
        )

    if X is not None:
        tmp = X[idx].permute(0, 2, 3, 4, 1).cpu().numpy()
        imgs = (tmp - tmp.min()) / (tmp.max() - tmp.min())
        imgs = np.clip(imgs, 0, 1).astype("float32")

        for t in range(T):
            wandb.log(
                {
                    f"{label}/img_org_t{t}": [
                        wandb.Image(
                            cv2.resize(im, output_dim),
                            caption="img_{}".format(i + 1),
                        )
                        for i, im in enumerate(imgs[:, t])
                    ],
                    f"global_{label}_step": global_step,
                }
            )
