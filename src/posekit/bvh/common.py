import numpy as np
from class_registry import AutoRegister, ClassRegistry, ClassRegistryInstanceCache

from posekit.skeleton import Skeleton, skeleton_registry
from posekit.video2bvh.bvh_skeleton import bvh_helper, math3d

_registry = ClassRegistry('name')
bvh_skeleton_registry = ClassRegistryInstanceCache(_registry)


# Uses code from https://github.com/KevinLTT/video2bvh.
class BvhSkeleton(metaclass=AutoRegister(_registry)):
    name = object()

    def __init__(self, bvh_joint_indices, bvh_joint_children, initial_directions, rotation_directions):
        self.bvh_joint_indices = bvh_joint_indices
        self.bvh_joint_children = bvh_joint_children
        self.initial_directions = initial_directions
        self.rotation_directions = rotation_directions

    @property
    def skeleton(self) -> Skeleton:
        return skeleton_registry[self.name]

    def get_bvh_header(self, poses_3d):
        initial_offset = self.calculate_initial_offset(poses_3d)
        root_joint_name = self.skeleton.joint_names[self.skeleton.root_joint_id]

        nodes = {}
        for joint in self.bvh_joint_indices:
            is_root = joint == root_joint_name
            is_end_site = joint.endswith('_endsite')
            nodes[joint] = bvh_helper.BvhNode(
                name=joint,
                offset=initial_offset[joint],
                rotation_order='zxy' if not is_end_site else '',
                is_root=is_root,
                is_end_site=is_end_site,
            )
        for joint, children in self.bvh_joint_children.items():
            nodes[joint].children = [nodes[child] for child in children]
            for child in children:
                nodes[child].parent = nodes[joint]

        header = bvh_helper.BvhHeader(root=nodes[root_joint_name], nodes=nodes)
        return header

    def pose2euler(self, pose, header):
        bvh_joint_parents = {}
        for parent, children in self.bvh_joint_children.items():
            for child in children:
                bvh_joint_parents[child] = parent

        channel = []
        quats = {}
        eulers = {}
        stack = [header.root]
        while stack:
            node = stack.pop()
            joint = node.name
            joint_idx = self.bvh_joint_indices[joint]

            if node.is_root:
                channel.extend(pose[joint_idx])

            if joint in self.rotation_directions:
                order, *pairs = self.rotation_directions[joint]
                dirs = []
                for pair in pairs:
                    if pair is None:
                        dirs.append(None)
                    else:
                        dirs.append(pose[self.bvh_joint_indices[pair[0]]] - pose[self.bvh_joint_indices[pair[1]]])
                x_dir, y_dir, z_dir = dirs
                dcm = math3d.dcm_from_axis(x_dir, y_dir, z_dir, order)
                quats[joint] = math3d.dcm2quat(dcm)
            else:
                quats[joint] = quats[bvh_joint_parents[joint]].copy()

            local_quat = quats[joint].copy()
            if node.parent:
                local_quat = math3d.quat_divide(quats[joint], quats[node.parent.name])

            euler = math3d.quat2euler(local_quat, node.rotation_order)
            euler = np.rad2deg(euler)
            eulers[joint] = euler
            channel.extend(euler)

            for child in node.children[::-1]:
                if not child.is_end_site:
                    stack.append(child)

        return channel

    def calculate_initial_offset(self, poses_3d):
        # TODO: Use RANSAC to better estimate bone lengths by rejecting outliers.
        root_joint_name = self.skeleton.joint_names[self.skeleton.root_joint_id]
        bone_lens = {root_joint_name: [0]}
        stack = [root_joint_name]
        while stack:
            parent = stack.pop()
            p_idx = self.bvh_joint_indices[parent]
            for child in self.bvh_joint_children[parent]:
                if self.bvh_joint_indices[child] < 0:
                    bone_lens[child] = 0.4 * bone_lens[parent]
                    continue
                stack.append(child)
                c_idx = self.bvh_joint_indices[child]
                bone_lens[child] = np.linalg.norm(
                    poses_3d[:, p_idx] - poses_3d[:, c_idx],
                    axis=1
                )

        bone_len = {}
        for joint in self.bvh_joint_indices:
            if joint.startswith('left_') or joint.startswith('right_'):
                base_name = joint.replace('left_', '').replace('right_', '')
                left_len = np.mean(bone_lens['left_' + base_name])
                right_len = np.mean(bone_lens['right_' + base_name])
                bone_len[joint] = (left_len + right_len) / 2
            else:
                bone_len[joint] = np.mean(bone_lens[joint])

        initial_offset = {}
        for joint, direction in self.initial_directions.items():
            direction = math3d.normalize(np.asarray(direction))
            initial_offset[joint] = direction * bone_len[joint]

        return initial_offset

    def save_poses(self, poses_3d, header=None, output_file=None, frame_rate=30):
        if not header:
            header = self.get_bvh_header(poses_3d)
        channels = [self.pose2euler(pose, header) for frame, pose in enumerate(poses_3d)]
        if output_file:
            bvh_helper.write_bvh(output_file, header, channels, frame_rate=frame_rate)
        return channels, header
