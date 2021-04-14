import os

from posekit.io.mocap import Mocap
from posekit.skeleton import skeleton_registry
from posekit.skeleton.utils import assert_plausible_skeleton
from posekit.video2bvh.bvh_skeleton.h36m_skeleton import H36mSkeleton


def save_bvh_mocap(mocap: Mocap, filename):
    if mocap.skeleton_name != 'h36m_17j':
        raise ValueError('only h36m_17j can be exported as BVH')
    skeleton = skeleton_registry[mocap.skeleton_name]
    assert_plausible_skeleton(mocap.joint_positions, skeleton)

    h36m_skel = H36mSkeleton()
    h36m_skel.poses2bvh(mocap.joint_positions / 1000, output_file=os.fspath(filename), frame_rate=mocap.sample_rate)
