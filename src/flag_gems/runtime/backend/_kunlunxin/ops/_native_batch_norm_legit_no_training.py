# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import torch

from ._batch_norm_no_update import _batch_norm_no_update

logger = logging.getLogger(__name__)


def _native_batch_norm_legit_no_training(
    input,
    weight=None,
    bias=None,
    running_mean=None,
    running_var=None,
    momentum=0.1,
    eps=1e-05,
):
    """Kunlunxin/XPU inference-only batch normalization using running stats.

    Mirrors ``torch.ops.aten._native_batch_norm_legit_no_training``. Delegates to
    the vendor _batch_norm_no_update, which uses the n_batch_groups fused kernel
    (grid = C*ceil(N/NB)) for small-spatial shapes — substantially faster than the
    old per-slice (grid = N*C) path on launch-bound shapes.
    Returns (output, save_mean, save_var) where save_mean/save_var are EMPTY
    (shape (0,)) since no batch statistics are computed in this mode.
    """
    logger.debug("GEMS_KUNLUNXIN _NATIVE_BATCH_NORM_LEGIT_NO_TRAINING")

    if running_mean is None or running_var is None:
        raise RuntimeError(
            "running_mean and running_var are required for "
            "_native_batch_norm_legit_no_training"
        )

    # _batch_norm_no_update returns (output, save_mean, save_var, reserved);
    # aten::_native_batch_norm_legit_no_training expects (output, save_mean, save_var).
    output, save_mean, save_var, _reserved = _batch_norm_no_update(
        input,
        weight=weight,
        bias=bias,
        running_mean=running_mean,
        running_var=running_var,
        momentum=momentum,
        eps=eps,
    )
    return output, save_mean, save_var
