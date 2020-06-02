import json

import numpy as np
import os

from glupy.math import ensure_cartesian
from posekit.io.mocap import Mocap
from posekit.skeleton import skeleton_registry
from posekit.skeleton.utils import assert_plausible_skeleton


_TEMPLATE = \
r'''{{
  "rate": {rate},
  "units": "mm",
  "skel": {skel},
  "points": {points}
}}'''


def save_json_mocap(mocap: Mocap, filename):
    skeleton = skeleton_registry[mocap.skeleton_name]
    assert_plausible_skeleton(mocap.joint_positions, skeleton)
    with open(os.fspath(filename), 'w') as f:
        joint_positions = ensure_cartesian(mocap.joint_positions, d=3)
        pose_blocks = []
        for pose in joint_positions:
            lines = [
                '      [{:0.6f}, {:0.6f}, {:0.6f}]'.format(*point)
                for point in pose
            ]
            pose_blocks.append(',\n'.join(lines))
        joined_pose_blocks = '\n    ], [\n'.join(pose_blocks)
        f.write(_TEMPLATE.format(
            rate=json.dumps(mocap.sample_rate),
            skel=json.dumps(skeleton.name),
            points=f'[\n    [\n{joined_pose_blocks}\n    ]\n  ]',
        ) + '\n')


def load_json_mocap(filename):
    with open(os.fspath(filename), 'r') as f:
        data = json.load(f)
    return Mocap(np.asarray(data['points'], dtype=np.float32), data['skel'], data['rate'])
