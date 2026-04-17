# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

def update_average(beta, old, new):
    if old is None:
        return new
    return old * beta + (1.0 - beta) * new


def update_moving_average(ma_model, current_model, beta):
    for current_params, ma_params in zip(
        current_model.parameters(), ma_model.parameters()
    ):
        old_weight, up_weight = ma_params.data, current_params.detach().data
        ma_params.data = update_average(beta, old_weight, up_weight)
