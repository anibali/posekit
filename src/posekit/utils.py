import numpy as np
import torch


def cast_array(values, ref):
    """Use a PyTorch/Numpy array representation for `values` that matches `ref`."""
    if torch.is_tensor(ref):
        return torch.as_tensor(values, dtype=ref.dtype)
    elif isinstance(ref, torch.dtype):
        return torch.as_tensor(values, dtype=ref)
    elif isinstance(ref, np.ndarray):
        return np.asarray(values, dtype=ref.dtype)
    elif np.issctype(ref):
        return np.asarray(values, dtype=ref)
    raise ValueError('`ref` must be a torch.Tensor, np.ndarray, or dtype')
