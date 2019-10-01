from dataclasses import dataclass
import numpy as np


@dataclass
class Mocap:
    joint_positions: np.ndarray
    skeleton_name: str
    sample_rate: float
