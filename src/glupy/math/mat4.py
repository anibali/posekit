import numpy as np


def identity():
    return np.eye(4, dtype=np.float64)


def affine(A=None, t=None):
    aff = identity()
    if A is not None:
        aff[0:3, 0:3] = np.array(A, dtype=aff.dtype)
    if t is not None:
        aff[0:3, 3] = np.array(t, dtype=aff.dtype)
    return aff


def flip_x(x=0):
    """Flip horizontally. x is the centre of reflection."""
    return affine(A=[[-1, 0, 0],
                     [ 0, 1, 0],
                     [ 0, 0, 1]],
                  t=[2 * x, 0, 0])


def translate(tx, ty, tz):
    """Translate."""
    return affine(t=[tx, ty, tz])


def rotate_quaternion(qr, qi, qj, qk):
    """Convert a quaternion into a rotation matrix."""
    return affine(A=[
        [1 - 2 * (qj**2 + qk**2),   2 * (qi * qj - qk * qr),    2 * (qi * qk + qj * qr)],
        [2 * (qi * qj + qk * qr),   1 - 2 * (qi**2 + qk**2),    2 * (qj * qk - qi * qr)],
        [2 * (qi * qk - qj * qr),   2 * (qj * qk + qi * qr),    1 - 2 * (qi**2 + qj**2)],
    ])


def rotate_axis_angle(ux, uy, uz, theta):
    """Rotate around an arbitrary axis. ux, uy, and uz must be components of a unit vector."""
    s = np.sin(theta / 2)
    return rotate_quaternion(np.cos(theta / 2), s * ux, s * uy, s * uz)


def scale(sx, sy=None, sz=None):
    """Scale."""
    if sy is None: sy = sx
    if sz is None: sz = sy
    return affine(A=[[sx,  0,  0],
                     [ 0, sy,  0],
                     [ 0,  0, sz]])


def perspective(fov_y, aspect, near, far):
    c = 1 / np.tan(fov_y / 2)
    return np.asarray([
        [c / aspect, 0, 0, 0],
        [0, -c, 0, 0],
        [0, 0, -(far + near) / (near - far), 2 * near * far / (near - far)],
        [0, 0, 1, 0],
    ])


def perspective_from_intrinsics(fx, fy, cx, cy, near, far):
    """Calculate an OpenGL perspective matrix from camera intrinsics.

    Note that the intrinsic parameters are expected to be normalised already. That is:
    * fx and cx should be divided by the image width.
    * fy and cy should be divided by the image height.

    Reference: https://blog.noctua-software.com/opencv-opengl-projection-matrix.html
    """
    return np.asarray([
        [2.0 * fx, 0, -(1.0 - 2.0 * cx), 0],
        [0, -2.0 * fy, 1.0 - 2.0 * cy, 0],
        [0, 0, -(far + near) / (near - far), 2 * near * far / (near - far)],
        [0, 0, 1, 0],
    ])


def orthographic(left, right, bottom, top, far=-1, near=1):
    return np.asarray([
        [2 / (right - left), 0, 0, -(right + left) / (right - left)],
        [0, 2 / (top - bottom), 0, -(top + bottom) / (top - bottom)],
        [0, 0, -2 / (far - near), -(far + near) / (far - near)],
        [0, 0, 0, 1],
    ])


def look_at(eye, target, up):
    camera_dir = target - eye
    camera_dir /= np.linalg.norm(camera_dir, 2)
    camera_right = np.cross(up, camera_dir)
    camera_right /= np.linalg.norm(camera_right)
    camera_up = np.cross(camera_dir, camera_right)
    view = np.eye(4)
    view[0, :3] = camera_right
    view[1, :3] = camera_up
    view[2, :3] = camera_dir
    view[:3, 3] = np.dot(view[:3, :3], -eye)
    return view


def concatenate(matrices):
    if len(matrices) == 1:
        return matrices[0]
    else:
        return concatenate(matrices[1:]) @ matrices[0]


def is_similarity(matrix, eps=1e-12):
    """Check whether the matrix represents a similarity transformation.

    Under a similarity transformation, relative lengths and angles remain unchanged.
    """
    A = matrix[:-1, :-1]
    v = matrix[-1]
    _, s, _ = np.linalg.svd(A)
    return np.max(s) - np.min(s) < eps \
           and (np.abs(v - np.asarray([0, 0, 0, 1], v.dtype)) < eps).all()
