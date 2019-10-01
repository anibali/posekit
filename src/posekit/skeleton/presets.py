from .common import Skeleton
from .converter import skeleton_converter
from .utils import move_joint_closer_, move_joint_farther_


class H36m17jSkeleton(Skeleton):
    name = 'h36m_17j'

    def __init__(self):
        super().__init__(
            joint_names=[
                'pelvis', 'right_hip', 'right_knee', 'right_ankle', 'left_hip',
                'left_knee', 'left_ankle', 'spine', 'neck', 'head',
                'head_top', 'left_shoulder', 'left_elbow', 'left_wrist', 'right_shoulder',
                'right_elbow', 'right_wrist'
            ],
            joint_tree=[
                0, 0, 1, 2, 0,
                4, 5, 0, 7, 8,
                8, 8, 11, 12, 8,
                14, 15
            ],
            hflip_indices=[
                0, 4, 5, 6, 1,
                2, 3, 7, 8, 9,
                10, 14, 15, 16, 11,
                12, 13
            ]
        )


class H36m32jSkeleton(Skeleton):
    name = 'h36m_32j'

    def __init__(self):
        super().__init__(
            joint_names=[
                'pelvis', 'right_hip', 'right_knee', 'right_ankle',
                'right_toes', 'right_site1', 'left_hip', 'left_knee',
                'left_ankle', 'left_toes', 'left_site1', 'spine1',
                'spine', 'neck', 'head', 'head_top',
                'left_clavicle', 'left_shoulder', 'left_elbow', 'left_wrist',
                'left_thumb', 'left_site2', 'left_wrist2', 'left_site3',
                'right_clavicle', 'right_shoulder', 'right_elbow', 'right_wrist',
                'right_thumb', 'right_site2', 'right_wrist2', 'right_site3'
            ],
            joint_tree=[
                0, 0, 1, 2,
                3, 4, 0, 6,
                7, 8, 9, 0,
                11, 12, 13, 14,
                12, 16, 17, 18,
                19, 20, 19, 22,
                12, 24, 25, 26,
                27, 28, 27, 30,
            ],
            hflip_indices=[
                0, 6, 7, 8,
                9, 10, 1, 2,
                3, 4, 5, 11,
                12, 13, 14, 15,
                24, 25, 26, 27,
                28, 29, 30, 31,
                16, 17, 18, 19,
                20, 21, 22, 23,
            ]
        )


class Mpii16jSkeleton(Skeleton):
    name = 'mpii_16j'

    def __init__(self):
        super().__init__(
            joint_names=[
                'right_ankle', 'right_knee', 'right_hip', 'left_hip',
                'left_knee', 'left_ankle', 'pelvis', 'spine',
                'neck', 'head_top', 'right_wrist', 'right_elbow',
                'right_shoulder', 'left_shoulder', 'left_elbow', 'left_wrist'
            ],
            joint_tree=[
                1, 2, 6, 6,
                3, 4, 6, 6,
                7, 8, 11, 12,
                8, 8, 13, 14
            ],
            hflip_indices=[
                5, 4, 3, 2,
                1, 0, 6, 7,
                8, 9, 15, 14,
                13, 12, 11, 10
            ]
        )


class Mpi3d17jSkeleton(Skeleton):
    name = 'mpi3d_17j'

    def __init__(self):
        super().__init__(
            joint_names=[
                'head_top', 'neck', 'right_shoulder', 'right_elbow', 'right_wrist',
                'left_shoulder', 'left_elbow', 'left_wrist', 'right_hip', 'right_knee',
                'right_ankle', 'left_hip', 'left_knee', 'left_ankle', 'pelvis',
                'spine', 'head'
            ],
            joint_tree=[
                1, 15, 1, 2, 3,
                1, 5, 6, 14, 8,
                9, 14, 11, 12, 14,
                14, 1
            ],
            hflip_indices=[
                0, 1, 5, 6, 7,
                2, 3, 4, 11, 12,
                13, 8, 9, 10, 14,
                15, 16
            ]
        )


class Mpi3d28jSkeleton(Skeleton):
    """28-joint skeleton from the MPI-INF-3DHP training data."""
    name = 'mpi3d_28j'

    def __init__(self):
        super().__init__(
            joint_names=[
                'spine3', 'spine4', 'spine2', 'spine',
                'pelvis', 'neck', 'head', 'head_top',
                'left_clavicle', 'left_shoulder', 'left_elbow', 'left_wrist',
                'left_hand', 'right_clavicle', 'right_shoulder', 'right_elbow',
                'right_wrist', 'right_hand', 'left_hip', 'left_knee',
                'left_ankle', 'left_foot', 'left_toe', 'right_hip',
                'right_knee', 'right_ankle', 'right_foot', 'right_toe'
            ],
            joint_tree=[
                2, 0, 3, 4,
                4, 1, 5, 6,
                5, 8, 9, 10,
                11, 5, 13, 14,
                15, 16, 4, 18,
                19, 20, 21, 4,
                23, 24, 25, 26
            ],
            hflip_indices=[
                0, 1, 2, 3,
                4, 5, 6, 7,
                13, 14, 15, 16,
                17, 8, 9, 10,
                11, 12, 23, 24,
                25, 26, 27, 18,
                19, 20, 21, 22
            ]
        )


def setup_canonical_skeleton_alias(aliased_skeleton_class):
    class CanonicalSkeleton(aliased_skeleton_class):
        name = 'canonical'

    @skeleton_converter.register(CanonicalSkeleton.name, aliased_skeleton_class.name)
    def convert_canonical_to_aliased(joints, from_skeleton, to_skeleton):
        return joints

    @skeleton_converter.register(aliased_skeleton_class.name, CanonicalSkeleton.name)
    def convert_aliased_to_canonical(joints, from_skeleton, to_skeleton):
        return joints


setup_canonical_skeleton_alias(H36m17jSkeleton)


def _subset_of_joints(joints, from_skeleton, joint_names):
    joint_indices = [from_skeleton.joint_index(s) for s in joint_names]
    return joints[..., joint_indices, :]


@skeleton_converter.register('h36m_32j', 'h36m_17j')
def convert_h36m_32j_to_h36m_17j(joints, from_skeleton, to_skeleton):
    return _subset_of_joints(joints, from_skeleton, to_skeleton.joint_names)


@skeleton_converter.register('mpi3d_28j', 'mpi3d_17j')
def convert_mpi3d_28j_to_mpi3d_17j(joints, from_skeleton, to_skeleton):
    return _subset_of_joints(joints, from_skeleton, to_skeleton.joint_names)


@skeleton_converter.register('mpii_16j', 'h36m_17j')
def convert_mpii_16j_to_h36m_17j(joints, from_skeleton, to_skeleton):
    joint_names = ['neck' if s == 'head' else s for s in to_skeleton.joint_names]
    dest_joints = _subset_of_joints(joints, from_skeleton, joint_names)

    # There is no 'head' joint in MPII, so we will interpolate between
    # 'head_top' and 'neck'. This is not a perfect solution, but it will have to do.
    move_joint_closer_(dest_joints, to_skeleton, 'head', 'head_top', 0.5)

    # The 'spine' joint in MPII is close to the neck, not in the middle of the back.
    # Therefore we need to move it closer to the pelvis.
    move_joint_closer_(dest_joints, to_skeleton, 'spine', 'pelvis', 0.47)

    return dest_joints


@skeleton_converter.register('h36m_17j', 'mpi3d_17j')
def convert_h36m_17j_to_mpi3d_17j(joints, from_skeleton, to_skeleton):
    dest_joints = _subset_of_joints(joints, from_skeleton, to_skeleton.joint_names)

    # Pelvis lift (pelvis is lower in Human3.6M compared to MPI-INF-3DHP).
    for joint_name in ['pelvis', 'left_hip', 'right_hip']:
        move_joint_closer_(dest_joints, to_skeleton, joint_name, 'spine', 0.07)

    return dest_joints


@skeleton_converter.register('mpi3d_17j', 'h36m_17j')
def convert_mpi3d_17j_to_h36m_17j(joints, from_skeleton, to_skeleton):
    dest_joints = _subset_of_joints(joints, from_skeleton, to_skeleton.joint_names)

    # Pelvis drop (pelvis is lower in Human3.6M compared to MPI-INF-3DHP).
    for joint_name in ['pelvis', 'left_hip', 'right_hip']:
        move_joint_farther_(dest_joints, to_skeleton, joint_name, 'spine', 0.07)

    return dest_joints