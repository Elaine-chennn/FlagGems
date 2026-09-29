import logging

import torch

logger = logging.getLogger(__name__)

# flag_gems registers vendor ops under this dispatch key (see
# runtime/op_registrar.py: reg_key = device.dispatch_key). On this XPU stack
# torch_xmlir maps the device to the CUDA dispatch key.
_BACKEND_DISPATCH_KEY = torch._C.DispatchKey.CUDA


def _conj(input: torch.Tensor) -> torch.Tensor:
    """Return the conjugate of a complex tensor.

    ``aten::_conj`` is a *view* op: it only toggles the tensor's conjugate bit
    (zero device work), exactly matching the ATen reference ``torch._conj``.
    Materializing the conjugate with a Triton kernel would enqueue full-tensor
    device traffic against a reference side that does nothing, capping speedup
    at the memory-bandwidth ceiling. We instead reproduce the native view.

    Under ``flag_gems.use_gems`` this function is itself the CUDA-key impl of
    ``aten::_conj``, so we exclude that key to redispatch to the native
    (Conjugate/composite) view implementation and avoid infinite recursion.
    When called directly (no use_gems active) the guard is a harmless no-op.
    """
    logger.debug("GEMS_KUNLUNXIN CONJ")
    if not input.is_complex():
        raise RuntimeError("_conj only supports complex tensors")

    with torch._C._ExcludeDispatchKeyGuard(
        torch._C.DispatchKeySet(_BACKEND_DISPATCH_KEY)
    ):
        return torch.ops.aten._conj.default(input)
