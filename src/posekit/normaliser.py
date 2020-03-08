"""Functions for transforming 3D points to and from a normalised space."""

from math import sqrt

import numpy as np
from posekit.camera import CameraIntrinsics
from posekit.utils import cast_array
from scipy import optimize


def camera_to_clip_matrix(z_ref, intrinsics, height, width):
    """Build a matrix that projects from camera space into clip space.

    Args:
        z_ref (float): The reference depth (will become z=0).
        intrinsics (CameraIntrinsics): The camera object specifying focal length and optical centre.
        height (float): The image height.
        width (float): The image width.

    Returns:
        np.ndarray: The projection matrix.
    """

    # Set the z-size (depth) of the viewing frustum to be equal to the
    # size of the portion of the XY plane at z_ref which projects
    # onto the image.
    size = z_ref * max(width / intrinsics.alpha_x, height / intrinsics.alpha_y)

    # Set near and far planes such that:
    # a) z_ref will correspond to z=0 after normalisation
    #    $z_ref = 2fn/(f+n)$
    # b) The distance from z=-1 to z=1 (normalised) will correspond
    #    to `size` in camera space
    #    $f - n = size$
    far = 0.5 * (sqrt(z_ref ** 2 + size ** 2) + z_ref - size)
    near = 0.5 * (sqrt(z_ref ** 2 + size ** 2) + z_ref + size)

    # Construct the perspective projection matrix.
    m_proj = cast_array([
        [intrinsics.alpha_x / intrinsics.x_0, 0, 0, 0],
        [0, intrinsics.alpha_y / intrinsics.y_0, 0, 0],
        [0, 0, -(far + near) / (far - near), 2 * far * near / (far - near)],
        [0, 0, 1, 0],
    ], intrinsics.matrix)

    return m_proj


def camera_space_to_ndc(Xc, P):
    """Transform point(s) from camera space to normalised device coordinates.

    Args:
        Xc (np.ndarray): homogeneous point(s) in camera space.
        P (np.ndarray): projection matrix.

    Returns:
        Xn (np.ndarray): homogeneous point(s) in normalised device coordinates.
    """
    # Camera space -> homogeneous clip space
    Xh = Xc @ P.T
    # Homogeneous clip space -> normalised device coordinates
    w = Xh[..., 3:4]
    Xn = Xh / w
    return Xn


def ndc_to_camera_space(Xn, P):
    """Transform point(s) from normalised device coordinates to camera space.

    Args:
        Xn (np.ndarray): homogeneous point(s) in normalised device coordinates.
        P (np.ndarray): projection matrix.

    Returns:
        Xc (np.ndarray): homogeneous point(s) in camera space.
    """
    # Normalised device coordinates -> homogeneous clip space
    z = Xn[..., 2:3]
    w = P[2, 3] / (z - P[2, 2])
    Xh = Xn * w
    # Homogeneous clip space -> camera space
    Xc = Xh @ np.linalg.inv(P).T
    return Xc


def normalise_points(denorm_points, z_ref, intrinsics, height, width):
    """Normalise a set of points, removing scale and z offset.

    Points within the image frame should have coordinate values between -1 and 1.

    Args:
        denorm_points (np.ndarray): The points to normalise.
        z_ref (float): The depth of the plane which will become z=0.
        intrinsics (CameraIntrinsics): The camera which projects 3D points onto the 2D image.
        height (float): The image height.
        width (float): The image width.

    Returns:
        np.ndarray: The normalised points.
    """
    m_proj = cast_array(camera_to_clip_matrix(z_ref, intrinsics, height, width), denorm_points)
    return camera_space_to_ndc(denorm_points, m_proj)


def denormalise_points(norm_points, z_ref, intrinsics, height, width):
    """Denormalise a set of points, adding scale and z offset.

    Args:
        norm_points (np.ndarray): The points to denormalise.
        z_ref (float): The depth of the plane that z=0 should map to.
        intrinsics (CameraIntrinsics): The camera which projects 3D points onto the 2D image.
        height (float): The image height.
        width (float): The image width.

    Returns:
        np.ndarray: The denormalised points.
    """
    m_proj = cast_array(camera_to_clip_matrix(z_ref, intrinsics, height, width), norm_points)
    return ndc_to_camera_space(norm_points, m_proj)


def infer_z_depth(norm_points, eval_scale, intrinsics, height, width, z_min=None, z_max=None):
    """Infer the depth of the root joint.

    Args:
        norm_points (np.ndarray): The normalised points.
        eval_scale (function): A function which evaluates the scale of denormalised points.
            This function is passed a proposed denormalised set of points, and is expected
            to return a scalar indicating how realistic the scale of the points is (<1.0 indicates
            too small, >1.0 indicates too large).
        intrinsics (CameraIntrinsics): The camera which projects 3D points onto the 2D image.
        height (float): The image height.
        width (float): The image width.
        z_min (float): Minimum bound for `z_ref`.
        z_max (float): Maximum bound for `z_ref`.

    Returns:
        float: `z_ref`, the depth of the root joint.
    """
    def f(z_ref):
        z_ref = float(z_ref)
        points = denormalise_points(norm_points, z_ref, intrinsics, height, width)
        k = eval_scale(points)
        return (k - 1.0) ** 2
    if z_min is None:
        z_min = max(intrinsics.alpha_x, intrinsics.alpha_y)
    if z_max is None:
        z_max = 20000
    z_ref = float(optimize.fminbound(f, z_min, z_max, maxfun=200, disp=0))
    return z_ref
