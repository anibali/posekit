from .common import BvhSkeleton


class H36m17jBvhSkeleton(BvhSkeleton):
    name = 'h36m_17j'

    def __init__(self):
        bvh_joint_indices = {}
        for i, joint_name in enumerate(self.skeleton.joint_names):
            if joint_name == 'head_top':
                joint_name = 'head_top_endsite'
            bvh_joint_indices[joint_name] = i
        bvh_joint_indices['right_ankle_endsite'] = -1
        bvh_joint_indices['left_ankle_endsite'] = -1
        bvh_joint_indices['left_wrist_endsite'] = -1
        bvh_joint_indices['right_wrist_endsite'] = -1

        index2keypoint = {}
        for k, v in bvh_joint_indices.items():
            index2keypoint[v] = k

        bvh_joint_children = {}
        for joint_name, i in bvh_joint_indices.items():
            joint_children = []
            if joint_name in self.skeleton.joint_names:
                for j, p in enumerate(self.skeleton.joint_tree):
                    if p == i and j != i:
                        joint_children.append(index2keypoint[j])
            bvh_joint_children[joint_name] = joint_children
        bvh_joint_children['right_ankle'] = ['right_ankle_endsite']
        bvh_joint_children['left_ankle'] = ['left_ankle_endsite']
        bvh_joint_children['right_wrist'] = ['right_wrist_endsite']
        bvh_joint_children['left_wrist'] = ['left_wrist_endsite']
        # Reparent head_top to head.
        bvh_joint_children['neck'] = ['head', 'left_shoulder', 'right_shoulder']
        bvh_joint_children['head'] = ['head_top_endsite']

        # Initial directions of joints (forms a T-pose).
        initial_directions = {
            'pelvis': [0, 0, 0],
            'right_hip': [-1, 0, 0],
            'right_knee': [0, 0, -1],
            'right_ankle': [0, 0, -1],
            'right_ankle_endsite': [0, -1, 0],
            'left_hip': [1, 0, 0],
            'left_knee': [0, 0, -1],
            'left_ankle': [0, 0, -1],
            'left_ankle_endsite': [0, -1, 0],
            'spine': [0, 0, 1],
            'neck': [0, 0, 1],
            'head': [0, 0, 1],
            'head_top_endsite': [0, 0, 1],
            'left_shoulder': [1, 0, 0],
            'left_elbow': [1, 0, 0],
            'left_wrist': [1, 0, 0],
            'left_wrist_endsite': [1, 0, 0],
            'right_shoulder': [-1, 0, 0],
            'right_elbow': [-1, 0, 0],
            'right_wrist': [-1, 0, 0],
            'right_wrist_endsite': [-1, 0, 0]
        }

        # Description of how to calculate joint rotations.
        # (rotation_order, ((to_x, from_x), (to_y, from_y), (to_z, from_z)))
        rotation_directions = {
            'pelvis':           ('zyx', ('left_hip', 'right_hip'), None, ('spine', 'pelvis')),
            'right_hip':        ('zyx', ('pelvis', 'right_hip'), None, ('right_hip', 'right_knee')),
            'right_knee':       ('zyx', ('pelvis', 'right_hip'), None, ('right_knee', 'right_ankle')),
            'left_hip':         ('zyx', ('left_hip', 'pelvis'), None, ('left_hip', 'left_knee')),
            'left_knee':        ('zyx', ('left_hip', 'pelvis'), None, ('left_knee', 'left_ankle')),
            'spine':            ('zyx', ('left_hip', 'right_hip'), None, ('neck', 'spine')),
            'neck':             ('zyx', ('left_shoulder', 'right_shoulder'), None, ('neck', 'spine')),
            # 'head':             ('zxy', None, ('neck', 'head'), ('head_top_endsite', 'neck')),
            'head':             ('zyx', ('left_shoulder', 'right_shoulder'), None, ('head_top_endsite', 'neck')),
            'left_shoulder':    ('xzy', ('left_elbow', 'left_shoulder'), ('left_elbow', 'left_wrist'), None),
            'left_elbow':       ('xzy', ('left_wrist', 'left_elbow'), ('left_elbow', 'left_shoulder'), None),
            'right_shoulder':   ('xzy', ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'), None),
            'right_elbow':      ('xzy', ('right_elbow', 'right_wrist'), ('right_elbow', 'right_shoulder'), None),
        }

        super().__init__(bvh_joint_indices, bvh_joint_children, initial_directions, rotation_directions)
