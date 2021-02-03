import csv
import os

import numpy as np

from glupy.math import ensure_cartesian
from posekit.io.mocap import Mocap
from posekit.skeleton import skeleton_registry
from posekit.skeleton.utils import assert_plausible_skeleton


def save_csv_mocap(mocap: Mocap, filename):
    skeleton = skeleton_registry[mocap.skeleton_name]
    assert_plausible_skeleton(mocap.joint_positions, skeleton)
    with open(os.fspath(filename), 'w') as f:
        fieldnames = ['t']
        for joint_name in skeleton.joint_names:
            for coord_name in 'xyz':
                fieldnames.append(f'{joint_name}_{coord_name}')
        writer = csv.writer(f)
        writer.writerow(fieldnames)

        joint_positions = ensure_cartesian(mocap.joint_positions, d=3)
        sample_period = 1 / mocap.sample_rate
        for i, pose in enumerate(joint_positions):
            row = [f'{i * sample_period:.4f}']
            for point in pose:
                for x in point:
                    row.append(f'{x:.6f}')
            writer.writerow(row)


def load_csv_mocap(filename):
    # CSV is expected to have the following structure for columns:
    # [t, JOINT1_x, JOINT1_y, JOINT1_z, JOINT2_x, ...]
    # We also assume that positions are in mm, and that t follows a constant sample rate.
    reader = csv.DictReader(open(os.fspath(filename), 'r'))
    joint_names = [header[:-2] for header in list(reader.fieldnames)[1::3]]
    for skeleton_name in skeleton_registry._registry.keys():
        if joint_names == skeleton_registry[skeleton_name].joint_names:
            break
    else:
        raise ValueError('failed to recognise skeleton structure')
    skeleton = skeleton_registry[skeleton_name]

    prev_t = None
    dts = []
    poses = []
    for row in reader:
        pose = np.zeros((skeleton.n_joints, 3), dtype=np.float32)
        for j, joint_name in enumerate(skeleton.joint_names):
            pose[j, 0] = row[f'{joint_name}_x']
            pose[j, 1] = row[f'{joint_name}_y']
            pose[j, 2] = row[f'{joint_name}_z']
        poses.append(pose)
        t = float(row['t'])
        if prev_t is not None:
            dts.append(t - prev_t)
        prev_t = t
    sample_rate = len(dts) / sum(dts)

    return Mocap(np.stack(poses), skeleton_name, sample_rate)
