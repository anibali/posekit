import csv

import numpy as np

from posekit.io.mocap import Mocap
from posekit.skeleton import skeleton_registry


def load_csv_mocap(filename):
    # CSV is expected to have the following structure for columns:
    # [t, JOINT1_x, JOINT1_y, JOINT1_z, JOINT2_x, ...]
    # We also assume that positions are in mm, and that t follows a constant sample rate.
    reader = csv.DictReader(open(filename))
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
        pose = np.zeros((skeleton.n_joints, 3))
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
