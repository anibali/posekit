import numpy as np
from glupy.utils import is_torch_tensor, is_torch_dtype, as_torch_tensor


def cast_array(values, ref):
    """Use a PyTorch/Numpy array representation for `values` that matches `ref`."""
    if is_torch_tensor(ref):
        return as_torch_tensor(values, dtype=ref.dtype)
    elif is_torch_dtype(ref):
        return as_torch_tensor(values, dtype=ref)
    elif isinstance(ref, np.ndarray):
        return np.asarray(values, dtype=ref.dtype)
    elif np.issctype(ref):
        return np.asarray(values, dtype=ref)
    raise ValueError('`ref` must be a torch.Tensor, np.ndarray, or dtype')
