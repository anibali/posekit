import ezc3d
import numpy as np
import os

from posekit.io.mocap import Mocap
from posekit.skeleton import skeleton_registry
from posekit.skeleton.utils import assert_plausible_skeleton


def save_c3d_mocap(mocap: Mocap, filename):
    skeleton = skeleton_registry[mocap.skeleton_name]
    assert_plausible_skeleton(mocap.joint_positions, skeleton)
    c3d = ezc3d.c3d()
    c3d['parameters']['POINT']['RATE']['value'] = [mocap.sample_rate]
    c3d['parameters']['POINT']['UNITS']['value'] = ['mm']
    c3d['parameters']['POINT']['LABELS']['value'] = skeleton.joint_names
    c3d.add_parameter('POINT', 'SKEL', [skeleton.name])
    c3d['data']['points'] = mocap.joint_positions.transpose(2, 1, 0).astype(np.float64)
    c3d.write(os.fspath(filename))


def load_c3d_mocap(filename):
    c3d = ezc3d.c3d(os.fspath(filename))
    sample_rate = c3d['parameters']['POINT']['RATE']['value'][0]
    skeleton_name = c3d['parameters']['POINT']['SKEL']['value'][0]
    joints = c3d['data']['points'].transpose(2, 1, 0)
    return Mocap(joints, skeleton_name, sample_rate)
