import numpy as np

from . import mat4


def to_cartesian(points):
    """Convert from homogeneous to cartesian coordinates."""
    return points[..., :-1] / points[..., -1:]


def to_homogeneous(points):
    """Convert from cartesian to homogeneous coordinates."""
    return np.concatenate([points, np.ones_like(points[..., -1:])], -1)


def rigid_registration(dest, src, reflection=False):
    """Rigid point set registration solver."""
    mtx1 = np.array(dest, dtype=np.double, copy=True)
    mtx2 = np.array(src, dtype=np.double, copy=True)

    mean1 = np.mean(mtx1, 0)
    mean2 = np.mean(mtx2, 0)
    mtx1 -= mean1
    mtx2 -= mean2

    norm1 = np.linalg.norm(mtx1)
    norm2 = np.linalg.norm(mtx2)
    mtx1 /= norm1
    mtx2 /= norm2

    u, w, vt = np.linalg.svd(mtx2.T.dot(mtx1).T)
    R = u.dot(vt)
    s = w.sum()

    if not reflection:
        # Determinant of R will be either -1 or 1. A value of -1 means that the alignment
        # opted to perform a reflection, which we will now undo.
        vt[2, :] *= np.linalg.det(R)
        R = u.dot(vt)

    T = mat4.concatenate([
        mat4.translate(*(-mean2)),
        mat4.scale(1 / norm2),
        mat4.affine(R),
        mat4.scale(norm1 * s),
        mat4.translate(*mean1),
    ])

    return T
