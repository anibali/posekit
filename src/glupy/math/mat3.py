import numpy as np


def identity():
    return np.eye(3, dtype=np.float64)


def affine(A=None, t=None):
    aff = identity()
    if A is not None:
        aff[0:2, 0:2] = A
    if t is not None:
        aff[0:2, 2] = t
    return aff


def flip_x(x=0):
    """Flip horizontally. x is the centre of reflection."""
    return affine(A=[[-1, 0],
                     [ 0, 1]],
                  t=[2 * x, 0])


def do_flip_x_(m, x=0):
    """Add a horizontal flip to a transformation matrix.

    Args:
        m: The 3x3 affine transformation matrix.
        x: The axis to flip about.
    """
    m[0] *= -1
    if x != 0:
        m[0, 2] += 2 * x


def rotate(theta):
    """Rotate counter-clockwise."""
    return affine(A=[[ np.cos(theta), np.sin(theta)],
                     [-np.sin(theta), np.cos(theta)]])


def scale(sx, sy=None):
    """Scale."""
    if sy is None: sy = sx
    return affine(A=[[sx,  0],
                     [ 0, sy]])


def do_scale_(m, sx, sy=None):
    """Add a scale transform to a transformation matrix.

    Args:
        m: The 3x3 affine transformation matrix.
        sx: x-axis scale factor.
        sy: y-axis scale factor.
    """
    if sy is None: sy = sx
    m[0] *= sx
    m[1] *= sy


def translate(tx, ty):
    """Translate."""
    return affine(t=[tx, ty])


def do_translate_(m, tx, ty):
    """Add a translation to a transformation matrix.

    Args:
        m: The 3x3 affine transformation matrix.
        tx: Translation in the x direction.
        ty: Translation in the y direction.
    """
    m[0, 2] += tx
    m[1, 2] += ty


def concatenate(matrices):
    if len(matrices) == 1:
        return matrices[0]
    else:
        return concatenate(matrices[1:]) @ matrices[0]
