import numpy as np
import pytest
import torch
from torch.testing import assert_allclose

from posekit.skeleton import skeleton_registry, skeleton_converter
from posekit.skeleton.utils import assert_plausible_skeleton, joints_to_kcs, move_joint_closer_, \
    absolute_to_root_relative, joints_to_limb_lengths, universal_orientation


def test_conversion_between_mpi3d_17j_and_h36m_17j():
    joints1 = torch.randn(4, 17, 2)
    joints2 = skeleton_converter.convert(joints1, 'mpi3d_17j', 'h36m_17j')
    joints3 = skeleton_converter.convert(joints2, 'h36m_17j', 'mpi3d_17j')
    assert_allclose(joints1, joints3)


@pytest.mark.parametrize('src_skeleton_name', skeleton_registry._registry.keys())
def test_convert_to_canonical(src_skeleton_name):
    if src_skeleton_name in {'coco_17j', 'coco_19j', 'vnect_14j'}:
        pytest.skip()
    skeleton = skeleton_registry[src_skeleton_name]
    joints = torch.randn(skeleton.n_joints, 3)
    canonical_joints = skeleton_converter.convert(joints, src_skeleton_name, 'canonical')
    assert canonical_joints.shape == (17, 3)


def test_assert_plausible_skeleton(annot2, skeleton):
    assert_plausible_skeleton(annot2, skeleton)


def test_joints_to_kcs():
    joints_3d = torch.ones((5, 28, 3), dtype=torch.float64)
    skeleton = skeleton_registry['mpi3d_28j']
    kcs = joints_to_kcs(joints_3d, skeleton)
    assert_allclose(kcs, torch.zeros((5, 27, 3), dtype=torch.float64))


def test_move_joint_closer(annot3, skeleton):
    movement_ratio = 0.371
    orig_left_ankle = annot3[skeleton.joint_index('left_ankle')]
    head_top = annot3[skeleton.joint_index('head_top')]
    annot3 = annot3.copy()
    move_joint_closer_(annot3, skeleton, 'left_ankle', 'head_top', movement_ratio)
    # Verify the change in distance between the joints.
    moved_left_ankle = annot3[skeleton.joint_index('left_ankle')]
    expected_dist = (1 - movement_ratio) * np.linalg.norm(orig_left_ankle - head_top)
    actual_dist = np.linalg.norm(moved_left_ankle - head_top)
    assert actual_dist == pytest.approx(expected_dist)


def test_absolute_to_root_relative(annot3, skeleton):
    new_annot3 = absolute_to_root_relative(annot3, skeleton)
    assert_allclose(new_annot3[skeleton.root_joint_id], 0)


def test_joints_to_limb_lengths(annot3, skeleton):
    actual = joints_to_limb_lengths(annot3, skeleton)
    expected = np.asarray([
        [266.5140], [265.0483], [147.8558], [323.4939], [243.6417], [157.9173], [323.4971],
        [243.6391], [124.9547], [523.6638], [418.2147], [124.9547], [521.9178], [418.2151],
        [3670.0883], [231.4023], [ 88.9872],
    ])
    assert_allclose(actual, expected)


def test_universal_orientation():
    joints_3d = np.array([
        [ 345.89, -211.17, 3735.89],
        [ 328.40,   36.50, 3751.33],
        [ 368.44,   96.13, 3630.29],
        [ 426.72,  168.51, 3340.30],
        [ 518.44,  183.03, 3126.30],
        [ 292.73,   87.88, 3879.10],
        [ 202.61,   78.87, 4169.83],
        [ 155.73,   67.08, 4398.05],
        [ 376.24,  506.35, 3667.88],
        [ 359.77,  990.50, 3726.00],
        [ 288.85, 1350.88, 3762.40],
        [ 301.87,  494.07, 3889.28],
        [ 308.86,  981.89, 3882.37],
        [ 259.36, 1345.81, 3845.72],
        [ 339.06,  500.21, 3778.58],
        [ 322.34,  281.36, 3761.52],
        [ 346.89,  -44.89, 3750.95],
    ])
    skeleton = skeleton_registry['mpi3d_17j']
    expected = np.array([
        [  -8.00, -654.92,  -18.64],
        [  -0.35, -426.92,    0.94],
        [-120.14, -377.62,    1.52],
        [-393.56, -325.00,   33.50],
        [-607.25, -320.95,   13.04],
        [ 119.30, -373.80,   -1.52],
        [ 399.52, -368.79,   -3.60],
        [ 613.02, -368.72,  -26.32],
        [-107.20,    1.35,   -1.63],
        [ -72.71,  448.83,   13.34],
        [ -36.66,  780.41,   77.82],
        [ 108.03,    0.08,    1.63],
        [  78.85,  448.02,   14.23],
        [  44.78,  779.49,   80.35],
        [   0.42,    0.71,   -0.00],
        [  -0.36, -201.56,   11.86],
        [  -2.31, -501.35,  -17.97],
    ]) / 375

    actual = np.array(universal_orientation(joints_3d, skeleton))
    assert_allclose(actual, expected, rtol=0, atol=0.1)
