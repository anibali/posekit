import os

from posekit.io.mocap import Mocap
from posekit.skeleton import skeleton_registry
from posekit.skeleton.utils import assert_plausible_skeleton
from posekit.bvh import BvhSkeleton, bvh_skeleton_registry


def save_bvh_mocap(mocap: Mocap, filename):
    skeleton = skeleton_registry[mocap.skeleton_name]
    assert_plausible_skeleton(mocap.joint_positions, skeleton)

    bvh_skeleton: BvhSkeleton = bvh_skeleton_registry[mocap.skeleton_name]
    poses_3d = mocap.joint_positions / 1000
    bvh_skeleton.save_poses(poses_3d, output_file=os.fspath(filename), frame_rate=mocap.sample_rate)
