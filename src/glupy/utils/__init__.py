def is_torch_tensor(obj):
    try:
        import torch
        return isinstance(obj, torch.Tensor)
    except ImportError:
        return False


def is_torch_dtype(obj):
    try:
        import torch
        return isinstance(obj, torch.dtype)
    except ImportError:
        return False


def as_torch_tensor(*args, **kwargs):
    import torch
    return torch.as_tensor(*args, **kwargs)
