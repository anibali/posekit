"""Functions for .trc (Track Row Column) input/output of motion capture data.

The .trc file format is useful for integration with OpenSim. For more information see:
https://simtk-confluence.stanford.edu:8443/display/OpenSim/Marker+%28.trc%29+Files
"""

import os
from copy import deepcopy

import numpy as np

from posekit.io.mocap import Mocap
from posekit.skeleton import skeleton_registry


def save_trc_mocap(mocap: Mocap, filename):
    skeleton = skeleton_registry[mocap.skeleton_name]
    n_frames = len(mocap.joint_positions)
    cols = 2 + skeleton.n_joints * 3
    def format_line(fields):
        assert len(fields) <= cols
        row = [str(e) for e in fields] + [''] * (cols - len(fields))
        return '\t'.join(row) + '\n'
    with open(os.fspath(filename), 'w') as f:
        f.write(format_line(['PathFileType', 4, '(X/Y/Z)', os.path.basename(filename)]))
        f.write(format_line(['DataRate', 'CameraRate', 'NumFrames', 'NumMarkers', 'Units',
                             'OrigDataRate', 'OrigDataStartFrame', 'OrigNumFrames']))
        f.write(format_line([mocap.sample_rate, mocap.sample_rate, n_frames, skeleton.n_joints,
                             'mm', mocap.sample_rate, 1, n_frames]))
        row = ['Frame#', 'Time']
        for joint_name in skeleton.joint_names:
            row.extend([joint_name, '', ''])
        f.write(format_line(row))
        row = ['', '']
        for i in range(1, len(skeleton.joint_names) + 1):
            row.extend([f'X{i}', f'Y{i}', f'Z{i}'])
        f.write(format_line(row))
        f.write(format_line([]))
        for i, pose_3d in enumerate(mocap.joint_positions[..., :3]):
            time = i / mocap.sample_rate
            f.write(format_line([i + 1, time] + list(pose_3d.flatten())))


def load_trc_mocap(filename):
    with open(os.fspath(filename), 'r') as f:
        lines = [line.strip().split('\t') for line in f.readlines() if line.strip()]
    metadata = {k: v for k, v in zip(lines[1], lines[2])}
    assert lines[2][4] == 'mm'
    # Infer the skeleton description from ordered joint names.
    joint_names = [e for e in lines[3][2:] if e]
    for skeleton_name, _ in skeleton_registry._registry.items():
        skeleton = skeleton_registry[skeleton_name]
        if skeleton.joint_names == joint_names:
            break
    else:
        raise ValueError('matching skeleton not found')
    # Load joint position data.
    poses = []
    for line in lines[5:]:
        pose = [float(x) for x in line[2:]]
        poses.append(np.reshape(pose, (skeleton.n_joints, 3)))
    assert len(poses) == int(metadata['NumFrames'])
    # Return a Mocap object representing the data.
    return Mocap(np.stack(poses), skeleton_name, int(metadata['DataRate']))


def save_opensim_trc_mocap(mocap: Mocap, filename):
    """Save an OpenSim-friendly .trc file.
    """
    mocap = deepcopy(mocap)
    skeleton = skeleton_registry[mocap.skeleton_name]
    ref_pos = mocap.joint_positions[0, skeleton.root_joint_id]
    # Make all joint locations relative to the initial root joint location.
    mocap.joint_positions -= ref_pos
    # Flip y and z axes.
    mocap.joint_positions[:, :, [1, 2]] *= -1
    save_trc_mocap(mocap, filename)
