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


def look_at(eye, target, up):
    camera_dir = target - eye
    camera_dir /= np.linalg.norm(camera_dir, 2)
    camera_right = np.cross(up, camera_dir)
    camera_up = np.cross(camera_dir, camera_right)
    view = np.eye(4)
    view[0, :3] = camera_right
    view[1, :3] = camera_up
    view[2, :3] = camera_dir
    view[:3, 3] = np.dot(view[:3, :3], -eye)
    return view
