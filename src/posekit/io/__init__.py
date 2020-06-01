import os
from pathlib import Path

from .c3d_mocap import load_c3d_mocap, save_c3d_mocap
from .csv_mocap import load_csv_mocap
from .json_mocap import load_json_mocap, save_json_mocap
from .mocap import Mocap

SUPPORTED_FORMATS = {'csv', 'c3d', 'json'}


def _infer_format(path, format):
    if format is None:
        format = path.suffix[1:]
    format = format.lower()
    if format not in SUPPORTED_FORMATS:
        raise ValueError(f'unrecognised mocap format: {format}')
    return format


def load_mocap(filename, format=None):
    path = Path(filename)
    format = _infer_format(path, format)

    if not path.is_file():
        raise FileNotFoundError(filename)

    if format == 'c3d':
        return load_c3d_mocap(os.fspath(path))
    elif format == 'csv':
        return load_csv_mocap(os.fspath(path))
    elif format == 'json':
        return load_json_mocap(os.fspath(path))
    raise NotImplementedError(f'loading {format}')


def save_mocap(mocap: Mocap, filename, format=None):
    path = Path(filename)
    format = _infer_format(path, format)

    if format == 'c3d':
        return save_c3d_mocap(mocap, os.fspath(path))
    elif format == 'json':
        return save_json_mocap(mocap, os.fspath(path))
    raise NotImplementedError(f'saving {format}')
