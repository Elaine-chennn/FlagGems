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

# NOTE (kunlunxin / XPU, 2026-09-15): why this file exists.
#
# The generic `flag_gems/ops/_native_batch_norm_legit.py` imports
# `batch_norm` from `flag_gems.ops.batch_norm` at MODULE IMPORT TIME, so the
# reference is closed over inside its __dict__ and the SpecOpRegistrar vendor
# override of `batch_norm` never reaches `_native_batch_norm_legit`.  On XPU
# the op therefore runs the GENERIC `batch_norm` Welford 2D-tile kernel, which
# fails to compile (batch_norm.py:107 `cnt += mask.to(tl.int32)` ->
# `triton_xpu.convert_layout` shape mismatch -> `TritonXPUUnrollControl` ->
# wrapped as `out of resource: uni_sram`).
#
# The fix is straightforward: implement the four _native_batch_norm_legit
# variants here, delegating to the vendor `native_batch_norm` kernels
# (same approach as _native_batch_norm_legit_functional.py).

import logging

from .native_batch_norm import native_batch_norm

logger = logging.getLogger(__name__)


def _native_batch_norm_legit(
    input,
    weight=None,
    bias=None,
    running_mean=None,
    running_var=None,
    training=False,
    momentum=0.1,
    eps=1e-5,
):
    """aten::_native_batch_norm_legit on Kunlunxin XPU.

    Delegates to the vendor native_batch_norm kernels instead of the generic
    batch_norm, which does not compile on this backend.
    """
    logger.debug("GEMS_KUNLUNXIN _NATIVE_BATCH_NORM_LEGIT")
    return native_batch_norm(
        input,
        weight=weight,
        bias=bias,
        running_mean=running_mean,
        running_var=running_var,
        training=training,
        momentum=momentum,
        eps=eps,
    )


def _native_batch_norm_legit_no_stats(
    input,
    weight=None,
    bias=None,
    training=True,
    momentum=0.1,
    eps=1e-5,
):
    """aten::_native_batch_norm_legit.no_stats on Kunlunxin XPU."""
    logger.debug("GEMS_KUNLUNXIN _NATIVE_BATCH_NORM_LEGIT_NO_STATS")
    import torch

    channels = input.shape[1]
    running_mean = torch.zeros(channels, dtype=input.dtype, device=input.device)
    running_var = torch.ones(channels, dtype=input.dtype, device=input.device)
    return native_batch_norm(
        input,
        weight=weight,
        bias=bias,
        running_mean=running_mean,
        running_var=running_var,
        training=True,
        momentum=momentum,
        eps=eps,
    )


def _copy_outputs(result, out, save_mean, save_invstd):
    result_out, result_mean, result_invstd = result
    out.resize_as_(result_out).copy_(result_out)
    save_mean.resize_as_(result_mean).copy_(result_mean)
    save_invstd.resize_as_(result_invstd).copy_(result_invstd)
    return out, save_mean, save_invstd


def _native_batch_norm_legit_out(
    input,
    weight=None,
    bias=None,
    running_mean=None,
    running_var=None,
    training=False,
    momentum=0.1,
    eps=1e-5,
    *,
    out,
    save_mean,
    save_invstd,
):
    """aten::_native_batch_norm_legit.out on Kunlunxin XPU."""
    logger.debug("GEMS_KUNLUNXIN _NATIVE_BATCH_NORM_LEGIT_OUT")
    result = _native_batch_norm_legit(
        input, weight, bias, running_mean, running_var, training, momentum, eps
    )
    return _copy_outputs(result, out, save_mean, save_invstd)


def _native_batch_norm_legit_no_stats_out(
    input,
    weight=None,
    bias=None,
    training=True,
    momentum=0.1,
    eps=1e-5,
    *,
    out,
    save_mean,
    save_invstd,
):
    """aten::_native_batch_norm_legit.no_stats_out on Kunlunxin XPU."""
    logger.debug("GEMS_KUNLUNXIN _NATIVE_BATCH_NORM_LEGIT_NO_STATS_OUT")
    result = _native_batch_norm_legit_no_stats(
        input, weight, bias, training, momentum, eps
    )
    return _copy_outputs(result, out, save_mean, save_invstd)
