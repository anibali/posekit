from abc import abstractmethod

import numpy as np
from class_registry import AutoRegister, ClassRegistry, ClassRegistryInstanceCache

_registry = ClassRegistry('name')
skeleton_registry = ClassRegistryInstanceCache(_registry)


class Skeleton(metaclass=AutoRegister(_registry)):
    name = object()

    @abstractmethod
    def __init__(self, joint_names, joint_tree, hflip_indices):
        """Description of a particular skeleton representation.

        Args:
            joint_names (list of str): Names of the joints.
            joint_tree (list of int): References to the parent of each joint.
            hflip_indices (list of int): References to the horizontal mirror of each joint.
        """
        assert len(joint_names) == len(joint_tree) == len(hflip_indices)
        self.joint_names = joint_names
        self.joint_tree = joint_tree
        self.hflip_indices = hflip_indices
        self.kcs_matrix_c = self._build_kcs_matrix_c()

    def _build_kcs_matrix_c(self):
        bone_descs = []
        for t, r in enumerate(self.joint_tree):
            if t != r:
                bone_desc = np.zeros(self.n_joints)
                bone_desc[t] = 1
                bone_desc[r] = -1
                bone_descs.append(bone_desc)
        return np.asarray(bone_descs)

    def joint_index(self, joint_name):
        return self.joint_names.index(joint_name)

    def topological_ordering(self):
        topo_joints = []
        nodes = [self.root_joint_id]
        while len(nodes) > 0:
            node = nodes.pop()
            for i in range(self.n_joints):
                if i != node and self.joint_tree[i] == node:
                    nodes.insert(0, i)
            topo_joints.append(node)
        return topo_joints

    @property
    def n_joints(self):
        """The number of joints in the skeleton."""
        return len(self.joint_names)

    @property
    def n_bones(self):
        """The number of bones in the skeleton."""
        return len(self.kcs_matrix_c)

    @property
    def root_joint_id(self):
        """The ID (index) of the root joint."""
        return self.joint_index('pelvis')

    def get_joint_metadata(self, joint_id):
        name = self.joint_names[joint_id]
        if name.startswith('left_'):
            group = 'left'
        elif name.startswith('right_'):
            group = 'right'
        else:
            group = 'centre'
        return dict(parent=self.joint_tree[joint_id], group=group)

    @property
    def leaf_joint_indices(self):
        """A list of indices for joints that do not have any children.
        """
        return [i for i in range(self.n_joints) if i not in self.joint_tree]
