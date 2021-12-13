from typing import Optional, Dict

import numpy as np

from glupy.utils import is_torch_tensor, as_torch_tensor
from . import skeleton_registry
from .utils import assert_plausible_skeleton


class _ConversionRegistry:
    def __init__(self):
        self._matrices = {}

    def register_matrix(self, from_skeleton_name: str, to_skeleton_name: str, mat: np.ndarray):
        """Register a conversion matrix.
        """
        from_skeleton = skeleton_registry[from_skeleton_name]
        to_skeleton = skeleton_registry[to_skeleton_name]
        if mat.shape != (from_skeleton.n_joints, to_skeleton.n_joints):
            raise ValueError(f'expected a {from_skeleton.n_joints} x {to_skeleton.n_joints} matrix')
        self._matrices.setdefault(from_skeleton_name, {})[to_skeleton_name] = mat

    def create_matrix(
        self,
        from_skeleton_name: str,
        to_skeleton_name: str,
        joint_map: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        """Create and register a new conversion matrix.
        """
        from_skeleton = skeleton_registry[from_skeleton_name]
        to_skeleton = skeleton_registry[to_skeleton_name]
        if joint_map is None:
            joint_map = {}
        mat = np.zeros((from_skeleton.n_joints, to_skeleton.n_joints))
        for j, joint_name in enumerate(to_skeleton.joint_names):
            if joint_name in joint_map:
                for src_joint_name, src_joint_weight in joint_map[joint_name].items():
                    mat[from_skeleton.joint_index(src_joint_name), j] = src_joint_weight
            elif joint_name in from_skeleton.joint_names:
                mat[from_skeleton.joint_index(joint_name), j] = 1.0
        self.register_matrix(from_skeleton_name, to_skeleton_name, mat)
        return mat

    def conversion_matrix(self, from_skeleton_name: str, to_skeleton_name: str):
        """Get a conversion matrix for mapping from one skeleton to another.

        If no direct conversion matrix has been defined previously, this function will attempt
        to compose conversion matrices.
        """
        if from_skeleton_name == to_skeleton_name:
            return np.eye(skeleton_registry[from_skeleton_name].n_joints)
        queue = list(self._matrices.get(from_skeleton_name, {}).items())
        seen = {from_skeleton_name}
        while len(queue) > 0:
            skeleton_name, mat = queue.pop()
            if skeleton_name == to_skeleton_name:
                return mat
            seen.add(skeleton_name)
            for next_skeleton_name, next_mat in self._matrices.get(skeleton_name, {}).items():
                if next_skeleton_name not in seen:
                    queue.insert(0, (next_skeleton_name, mat @ next_mat))
        raise NotImplementedError(f'no skeleton conversion path defined from {from_skeleton_name} to {to_skeleton_name}')

    def convert(self, joints, from_skeleton_name: str, to_skeleton_name: str):
        """Convert joint keypoints from one skeleton to another.
        """
        assert_plausible_skeleton(joints, skeleton_registry[from_skeleton_name])
        mat = self.conversion_matrix(from_skeleton_name, to_skeleton_name)
        new_joints = np.einsum('...ij,ik->...kj', np.asarray(joints), mat)
        if is_torch_tensor(joints):
            new_joints = as_torch_tensor(new_joints).to(joints)
        return new_joints


skeleton_converter = _ConversionRegistry()
