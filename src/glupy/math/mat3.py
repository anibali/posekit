import numpy as np


def identity():
    return np.eye(3, dtype=np.float64)


def affine(A=None, t=None):
    aff = identity()
    if A is not None:
        aff[0:2, 0:2] = np.array(A, dtype=aff.dtype)
    if t is not None:
        aff[0:2, 2] = np.array(t, dtype=aff.dtype)
    return aff


def flip_x(x=0):
    """Flip horizontally. x is the centre of reflection."""
    return affine(A=[[-1, 0],
                     [ 0, 1]],
                  t=[2 * x, 0])


def rotate(theta):
    """Rotate counter-clockwise."""
    return affine(A=[[ np.cos(theta), np.sin(theta)],
                     [-np.sin(theta), np.cos(theta)]])


def scale(sx, sy=None):
    """Scale."""
    if sy is None: sy = sx
    return affine(A=[[sx,  0],
                     [ 0, sy]])


def translate(tx, ty):
    """Translate."""
    return affine(t=[tx, ty])


def concatenate(matrices):
    if len(matrices) == 1:
        return matrices[0]
    else:
        return concatenate(matrices[1:]) @ matrices[0]
