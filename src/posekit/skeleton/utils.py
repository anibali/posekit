import numpy as np
import torch

from glupy.math import point_set_registration, to_cartesian, to_homogeneous
from .common import Skeleton


def assert_plausible_skeleton(joints, skeleton: Skeleton):
    assert joints.shape[-2] == skeleton.n_joints


def move_joint_closer_(joints, skeleton: Skeleton, joint_name, other_joint_name, alpha):
    joints[..., skeleton.joint_index(joint_name), :] *= (1 - alpha)
    joints[..., skeleton.joint_index(joint_name), :] += alpha * joints[..., skeleton.joint_index(other_joint_name), :]


def move_joint_farther_(joints, skeleton: Skeleton, joint_name, other_joint_name, alpha):
    joints[..., skeleton.joint_index(joint_name), :] -= alpha * joints[..., skeleton.joint_index(other_joint_name), :]
    joints[..., skeleton.joint_index(joint_name), :] /= (1 - alpha)


def procrustes(ref_points, cor_points, points=None, *, reflection=False):
    """Align `cor_points` to `ref_points`, and apply the resulting transformation to `points`.

    Args:
        ref_points: Reference points.
        cor_points: Observed points corresponding to reference points.
        points: Points to transform according to the discovered alignment.
                Defaults to `cor_points`.
        reflection: If set to `True`, permit reflections in the transform.

    Returns:
        (np.ndarray) The transformed points.
    """

    T = point_set_registration(ref_points, cor_points, reflection)
    return to_cartesian(to_homogeneous(points) @ T.T)


def absolute_to_root_relative(joints: torch.Tensor, skeleton: Skeleton):
    assert_plausible_skeleton(joints, skeleton)
    root_pos = joints[..., skeleton.root_joint_id:skeleton.root_joint_id+1, :]
    return joints - root_pos


def absolute_to_parent_relative(joints: torch.Tensor, skeleton: Skeleton):
    return joints - joints[..., skeleton.joint_tree, :]


def joints_to_kcs(joints, skeleton: Skeleton):
    assert_plausible_skeleton(joints, skeleton)
    C = torch.tensor(skeleton.kcs_matrix_c, dtype=joints.dtype, device=joints.device)
    return torch.matmul(C, joints)


def joints_to_limb_lengths(joints, skeleton: Skeleton):
    cartesian = absolute_to_parent_relative(joints, skeleton)
    r = (cartesian ** 2).sum(-1)[..., None] ** 0.5
    return r


UNIVERSAL_TORSO = np.array([
    [  0.317, -1.0, 0.0], # Left shoulder
    [ -0.317, -1.0, 0.0], # Right shoulder
    [  0.296,  0.0, 0.0], # Left hip
    [ -0.296,  0.0, 0.0], # Right hip
])


def universal_orientation(joints_3d, skeleton: Skeleton):
    """Apply a transformation to the skeleton such that it has a canonical orientation.

    From the perspective of the torso, directions are:
    * Front/forwards: negative Z
    * Right: negative X
    * Up: negative Y

    Furthermore, the skeleton will be centred such that the root joint is at (0,0,0)
    """
    torso = np.stack([
        joints_3d[skeleton.joint_names.index('left_shoulder')],
        joints_3d[skeleton.joint_names.index('right_shoulder')],
        joints_3d[skeleton.joint_names.index('left_hip')],
        joints_3d[skeleton.joint_names.index('right_hip')],
    ])
    new_joints_3d = procrustes(UNIVERSAL_TORSO, torso, joints_3d, reflection=False)
    new_joints_3d -= new_joints_3d[skeleton.root_joint_id]
    return new_joints_3d


def flip_joints(joints, skeleton: Skeleton):
    joints = joints[..., skeleton.hflip_indices, :]
    joints[..., 0] *= -1
    return joints


def average_flipped_joints(joints, flipped_joints, skeleton: Skeleton):
    return (joints + flip_joints(flipped_joints, skeleton)) / 2
