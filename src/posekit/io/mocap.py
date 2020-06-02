from dataclasses import dataclass
import numpy as np


@dataclass
class Mocap:
    # A n_frames x n_joints x 3 tensor containing 3D cartesian coordinates of each joint at each
    # time instant. Unit: mm.
    joint_positions: np.ndarray
    # The name of the skeleton description for the joints in `joint_positions`.
    skeleton_name: str
    # The frame sample rate. Unit: Hz.
    sample_rate: float
