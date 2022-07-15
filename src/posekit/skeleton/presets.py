from typing import Dict, Optional

import numpy as np

from .common import Skeleton, skeleton_registry
from .converter import skeleton_converter


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


# Official PoseTrack keypoint locations. The pelvis joint is missing so there isn't a good root
# joint choice (neck is used here).
class PoseTrack15jSkeleton(Skeleton):
    name = 'posetrack_15j'

    def __init__(self):
        super().__init__(
            joint_names=[
                'right_ankle', 'right_knee', 'right_hip',
                'left_hip', 'left_knee', 'left_ankle',
                'right_wrist', 'right_elbow', 'right_shoulder',
                'left_shoulder', 'left_elbow', 'left_wrist',
                'neck', 'nose', 'head_top',
            ],
            joint_tree=[
                1, 2, 12,
                12, 3, 4,
                7, 8, 12,
                12, 9, 10,
                12, 12, 12,
            ],
            hflip_indices=[
                5, 4, 3,
                2, 1, 0,
                11, 10, 9,
                8, 7, 6,
                12, 13, 14,
            ]
        )

    def root_joint_id(self):
        return 12


# Unofficial PoseTrack keypoint locations with the pelvis joint added in.
class PoseTrack16jSkeleton(Skeleton):
    name = 'posetrack_16j'

    def __init__(self):
        super().__init__(
            joint_names=[
                'right_ankle', 'right_knee', 'right_hip',
                'left_hip', 'left_knee', 'left_ankle',
                'right_wrist', 'right_elbow', 'right_shoulder',
                'left_shoulder', 'left_elbow', 'left_wrist',
                'neck', 'nose', 'head_top', 'pelvis',
            ],
            joint_tree=[
                1, 2, 15,
                15, 3, 4,
                7, 8, 12,
                12, 9, 10,
                15, 12, 12,
                15,
            ],
            hflip_indices=[
                5, 4, 3,
                2, 1, 0,
                11, 10, 9,
                8, 7, 6,
                12, 13, 14,
                15,
            ]
        )


# Official COCO keypoint locations. This is a problematic representation since it does not
# define a proper joint tree, and is missing the root joint.
class Coco17jSkeleton(Skeleton):
    name = 'coco_17j'

    def __init__(self):
        super().__init__(
            joint_names=[
                'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist',
                'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee',
                'left_ankle', 'right_ankle'
            ],
            joint_tree=[
                0, 0, 0, 1, 2,
                0, 0, 5, 6, 7,
                8, 0, 0, 11, 12,
                13, 14
            ],
            hflip_indices=[
                0, 2, 1, 4, 3,
                6, 5, 8, 7, 10,
                9, 12, 11, 14, 13,
                16, 15
            ]
        )

    def root_joint_id(self):
        return 0


# This is an unofficial set of COCO keypoints which adds neck and pelvis joints, thus making the
# skeleton a valid hierarchy including a root joint.
class Coco19jSkeleton(Skeleton):
    name = 'coco_19j'

    def __init__(self):
        super().__init__(
            joint_names=[
                'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist',
                'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee',
                'left_ankle', 'right_ankle', 'pelvis', 'neck'
            ],
            joint_tree=[
                18, 0, 0, 1, 2,
                18, 18, 5, 6, 7,
                8, 17, 17, 11, 12,
                13, 14, 17, 17
            ],
            hflip_indices=[
                0, 2, 1, 4, 3,
                6, 5, 8, 7, 10,
                9, 12, 11, 14, 13,
                16, 15, 17, 18
            ]
        )


# COCO keypoint locations as used by OpenPose.
# See:
# * https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/1f1aa9c59fe59c90cca685b724f4f97f76137224/doc/02_output.md#pose-output-format-coco
# * https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/16ab72801d9ebfee12903a6678b0e41e7956cb7e/src/openpose/pose/poseParameters.cpp#L35-L55
class OpenPose18jSkeleton(Skeleton):
    name = 'openpose_18j'

    def __init__(self):
        super().__init__(
            joint_names=[
                'nose', 'neck', 'right_shoulder', 'right_elbow', 'right_wrist',
                'left_shoulder', 'left_elbow', 'left_wrist', 'right_hip', 'right_knee',
                'right_ankle', 'left_hip', 'left_knee', 'left_ankle', 'right_eye',
                'left_eye', 'right_ear', 'left_ear',
            ],
            joint_tree=[
                1, 1, 1, 2, 3,
                1, 5, 6, 1, 8,
                9, 1, 11, 12, 0,
                0, 14, 15,
            ],
            hflip_indices=[
                0, 0, 5, 6, 7,
                2, 3, 4, 11, 12,
                13, 8, 9, 10, 15,
                14, 17, 16,
            ]
        )

    def root_joint_id(self):
        return self.joint_index('neck')


# OpenPose (25-joint).
# See:
# * https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/1f1aa9c59fe59c90cca685b724f4f97f76137224/doc/02_output.md#pose-output-format-body_25
# * https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/16ab72801d9ebfee12903a6678b0e41e7956cb7e/src/openpose/pose/poseParameters.cpp#L7-L34
class OpenPose25jSkeleton(Skeleton):
    name = 'openpose_25j'

    def __init__(self):
        super().__init__(
            joint_names=[
                'nose', 'neck', 'right_shoulder', 'right_elbow', 'right_wrist',
                'left_shoulder', 'left_elbow', 'left_wrist', 'pelvis', 'right_hip',
                'right_knee', 'right_ankle', 'left_hip', 'left_knee', 'left_ankle',
                'right_eye', 'left_eye', 'right_ear', 'left_ear', 'left_big_toe',
                'left_small_toe', 'left_heel', 'right_big_toe', 'right_small_toe', 'right_heel',
            ],
            joint_tree=[
                1, 8, 1, 2, 3,
                1, 5, 6, 8, 8,
                9, 10, 8, 12, 13,
                0, 0, 15, 16, 14,
                19, 14, 11, 22, 11,
            ],
            hflip_indices=[
                0, 1, 5, 6, 7,
                2, 3, 4, 8, 12,
                13, 14, 9, 10, 11,
                16, 15, 18, 17, 22,
                23, 24, 19, 20, 21,
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

# Subset of mpi3d_28j used for evaluation in the VNect paper.
class Vnect14jSkeleton(Skeleton):
    name = 'vnect_14j'

    def __init__(self):
        super().__init__(
            joint_names=[
                'head_top', 'neck', 'right_shoulder', 'right_elbow', 'right_wrist',
                'left_shoulder', 'left_elbow', 'left_wrist', 'right_hip', 'right_knee',
                'right_ankle', 'left_hip', 'left_knee', 'left_ankle'
            ],
            joint_tree=[
                1, 1, 1, 2, 3,
                1, 5, 6, 1, 8,
                9, 1, 11, 12
            ],
            hflip_indices=[
                0, 1, 5, 6, 7,
                2, 3, 4, 11, 12,
                13, 8, 9, 10
            ]
        )


class Aspset17jSkeleton(Skeleton):
    name = 'aspset_17j'

    def __init__(self):
        super().__init__(
            joint_names=[
                'right_ankle',      'right_knee',       'right_hip',
                'right_wrist',      'right_elbow',      'right_shoulder',
                'left_ankle',       'left_knee',        'left_hip',
                'left_wrist',       'left_elbow',       'left_shoulder',
                'head_top',         'head',             'neck',
                'spine',            'pelvis',
            ],
            joint_tree=[
                1, 2, 16, 4, 5, 14,
                7, 8, 16, 10, 11, 14,
                13, 14, 15, 16, 16,
            ],
            hflip_indices=[
                6, 7, 8, 9, 10, 11,
                0, 1, 2, 3, 4, 5,
                12, 13, 14, 15, 16,
            ]
        )


class Aspset21jSkeleton(Skeleton):
    name = 'aspset_21j'

    def __init__(self):
        super().__init__(
            joint_names=[
                'right_ankle',      'right_knee',       'right_hip',
                'right_wrist',      'right_elbow',      'right_shoulder',
                'left_ankle',       'left_knee',        'left_hip',
                'left_wrist',       'left_elbow',       'left_shoulder',
                'head_top',         'head',             'neck',
                'spine',            'pelvis',           'right_toe_base',
                'right_heel',       'left_toe_base',    'left_heel',
            ],
            joint_tree=[
                1, 2, 16, 4, 5, 14,
                7, 8, 16, 10, 11, 14,
                13, 14, 15, 16, 16,
                18, 0, 20, 6,
            ],
            hflip_indices=[
                6, 7, 8, 9, 10, 11,
                0, 1, 2, 3, 4, 5,
                12, 13, 14, 15, 16,
                19, 20, 17, 18,
            ]
        )


class AspsetFeet4jSkeleton(Skeleton):
    name = 'aspsetfeet_4j'

    def __init__(self):
        super().__init__(
            joint_names=[
                'right_toe_base', 'right_heel', 'left_toe_base', 'left_heel',
            ],
            joint_tree=[
                1, 1, 3, 3
            ],
            hflip_indices=[
                2, 3, 0, 1
            ]
        )


class Lsp14jSkeleton(Skeleton):
    name = 'lsp_14j'

    def __init__(self):
        super().__init__(
            joint_names=[
                'right_ankle', 'right_knee', 'right_hip',
                'left_hip', 'left_knee', 'left_ankle',
                'right_wrist', 'right_elbow', 'right_shoulder',
                'left_shoulder', 'left_elbow', 'left_wrist',
                'neck', 'head_top',
            ],
            joint_tree=[
                1, 2, 12,
                12, 3, 4,
                7, 8, 12,
                12, 9, 10,
                12, 12
            ],
            hflip_indices=[
                5, 4, 3,
                2, 1, 0,
                11, 10, 9,
                8, 7, 6,
                12, 13
            ]
        )


# This is an unofficial set of LSP keypoints which adds a pelvis joint, thus making the
# skeleton a valid hierarchy including a root joint.
class Lsp15jSkeleton(Skeleton):
    name = 'lsp_15j'

    def __init__(self):
        super().__init__(
            joint_names=[
                'right_ankle', 'right_knee', 'right_hip',
                'left_hip', 'left_knee', 'left_ankle',
                'right_wrist', 'right_elbow', 'right_shoulder',
                'left_shoulder', 'left_elbow', 'left_wrist',
                'neck', 'head_top', 'pelvis',
            ],
            joint_tree=[
                1, 2, 14,
                14, 3, 4,
                7, 8, 12,
                12, 9, 10,
                14, 12, 14,
            ],
            hflip_indices=[
                5, 4, 3,
                2, 1, 0,
                11, 10, 9,
                8, 7, 6,
                12, 13, 14
            ]
        )


# Joints for the SMPL body model.
# See https://github.com/gulvarol/surreal/blob/master/datageneration/misc/smpl_relations/smpl_relations.py
# and Fig. 3. from "Hierarchical Kinematic Human Mesh Recovery".
class Smpl24jSkeleton(Skeleton):
    name = 'smpl_24j'

    def __init__(self):
        super().__init__(
            joint_names=[
                'pelvis', 'left_hip', 'right_hip',
                'spine', 'left_knee', 'right_knee',
                'spine1', 'left_ankle', 'right_ankle',
                'spine2', 'left_foot', 'right_foot',
                'neck', 'left_collar', 'right_collar',
                'head', 'left_shoulder', 'right_shoulder',
                'left_elbow', 'right_elbow', 'left_wrist',
                'right_wrist', 'left_hand', 'right_hand',
            ],
            joint_tree=[
                0, 0, 0,
                0, 1, 2,
                3, 4, 5,
                6, 7, 8,
                9, 12, 12,
                12, 13, 14,
                16, 17, 18,
                19, 20, 21,
            ],
            hflip_indices=[
                0, 2, 1,
                3, 5, 4,
                6, 8, 7,
                9, 11, 10,
                12, 14, 13,
                15, 17, 16,
                19, 18, 21,
                20, 23, 22,
            ]
        )


def _setup_canonical_skeleton_alias(aliased_skeleton_class):
    class CanonicalSkeleton(aliased_skeleton_class):
        name = 'canonical'
    skeleton_converter.create_matrix(CanonicalSkeleton.name, aliased_skeleton_class.name)
    skeleton_converter.create_matrix(aliased_skeleton_class.name, CanonicalSkeleton.name)


_setup_canonical_skeleton_alias(H36m17jSkeleton)


def _tweak_joint_position(skeleton: Skeleton, joint_name: str, other_joint_name: str, factor: float):
    """Create a matrix which tweaks a joint position.

    Args:
        skeleton: The skeleton definition.
        joint_name: The joint name.
        other_joint_name: The other joint name.
        factor: The mixing factor. If set to 1.0, the joint will be unaffected.
            If set to 0.0, the joint will copy the other joint perfectly. Note that the tweak can
            be undone by applying another tweak with 1 / factor.

    Returns:
        A matrix for applying the tweak.
    """
    m = np.eye(skeleton.n_joints)
    m[skeleton.joint_index(joint_name), skeleton.joint_index(joint_name)] = factor
    m[skeleton.joint_index(other_joint_name), skeleton.joint_index(joint_name)] = 1 - factor
    return m


def _create_conversion_matrices():
    skeleton_converter.create_matrix('posetrack_15j', 'posetrack_16j', joint_map={
        'pelvis': {'left_hip': 0.5, 'right_hip': 0.5},
    })

    skeleton_converter.create_matrix('posetrack_16j', 'posetrack_15j')

    skeleton_converter.create_matrix('posetrack_16j', 'h36m_17j', joint_map={
        'spine': {'pelvis': 0.629, 'neck': 1 - 0.629},
        'head': {'nose': 1.0},
    })

    skeleton_converter.create_matrix('h36m_17j', 'posetrack_16j', joint_map={
        'nose': {'head': 1.0},
    })

    skeleton_converter.create_matrix('h36m_32j', 'h36m_17j')

    skeleton_converter.create_matrix('mpi3d_28j', 'mpi3d_17j')

    skeleton_converter.create_matrix('mpi3d_17j', 'vnect_14j')

    skeleton_converter.create_matrix('mpii_16j', 'h36m_17j', joint_map={
        # There is no 'head' joint in MPII, so we will interpolate between
        # 'head_top' and 'neck'. This is not a perfect solution, but it will have to do.
        'head': {'neck': 0.5, 'head_top': 0.5},
        # The 'spine' joint in MPII is close to the neck, not in the middle of the back.
        # Therefore we need to move it closer to the pelvis.
        'spine': {'spine': 0.53, 'pelvis': 1 - 0.53},
    })

    skeleton_converter.create_matrix('h36m_17j', 'mpii_16j', joint_map={
        'spine': {'spine': 1 / 0.53, 'pelvis': 1 - 1 / 0.53},
    })

    skeleton_converter.create_matrix('h36m_17j', 'mpi3d_17j', joint_map={
        # Pelvis lift (pelvis is lower in Human3.6M compared to MPI-INF-3DHP).
        'pelvis': {'pelvis': 0.93, 'spine': 1 - 0.93},
        'left_hip': {'left_hip': 0.93, 'spine': 1 - 0.93},
        'right_hip': {'right_hip': 0.93, 'spine': 1 - 0.93},
    })

    skeleton_converter.create_matrix('mpi3d_17j', 'h36m_17j', joint_map={
        # Pelvis drop (pelvis is lower in Human3.6M compared to MPI-INF-3DHP).
        'pelvis': {'pelvis': 1 / 0.93, 'spine': 1 - 1 / 0.93},
        'left_hip': {'left_hip': 1 / 0.93, 'spine': 1 - 1 / 0.93},
        'right_hip': {'right_hip': 1 / 0.93, 'spine': 1 - 1 / 0.93},
    })

    skeleton_converter.create_matrix('coco_17j', 'coco_19j', joint_map={
        'pelvis': {'left_hip': 0.5, 'right_hip': 0.5},
        'neck': {'left_shoulder': 0.5, 'right_shoulder': 0.5},
    })

    skeleton_converter.create_matrix('coco_19j', 'coco_17j')

    skeleton_converter.create_matrix('openpose_25j', 'openpose_18j')

    skeleton_converter.create_matrix('openpose_18j', 'coco_19j', joint_map={
        'pelvis': {'left_hip': 0.5, 'right_hip': 0.5},
    })

    skeleton_converter.create_matrix('coco_19j', 'openpose_18j')

    c = 0.6
    mat = skeleton_converter.create_matrix('coco_19j', 'h36m_17j', joint_map={
        'spine': {'pelvis': 0.629, 'neck': (1 - 0.629)},
        'head': {'nose': 1.0},
        'head_top': {'left_ear': (1 + c) * 0.5, 'right_ear': (1 + c) * 0.5, 'neck': -c},
    })
    np.matmul(mat, _tweak_joint_position(skeleton_registry['h36m_17j'], 'neck', 'head_top', 0.8), out=mat)

    skeleton_converter.create_matrix('aspset_21j', 'aspset_17j')

    skeleton_converter.create_matrix('aspset_21j', 'aspsetfeet_4j')

    skeleton_converter.create_matrix('mpii_16j', 'aspset_17j', joint_map={
        'head': {'head_top': 0.33, 'neck': 1 - 0.67},
        'pelvis': {'pelvis': 0.84, 'spine': 0.48, 'neck': -0.32},
        'spine': {'spine': 0.66, 'pelvis': 0.53, 'neck': -0.20}
    })

    skeleton_converter.create_matrix('aspset_17j', 'mpii_16j', joint_map={
        'pelvis': {'left_hip': 0.5, 'right_hip': 0.5},
        'spine': {'spine': -0.14, 'pelvis': 0.25, 'neck': 0.89},
    })

    skeleton_converter.create_matrix('lsp_14j', 'lsp_15j', joint_map={
        'pelvis': {'left_hip': 0.5, 'right_hip': 0.5},
    })

    skeleton_converter.create_matrix('lsp_15j', 'lsp_14j')

    skeleton_converter.create_matrix('lsp_15j', 'mpii_16j', joint_map={
        'spine': {'neck': 0.8, 'pelvis': 1 - 0.8},
    })

    skeleton_converter.create_matrix('mpii_16j', 'lsp_15j')


_create_conversion_matrices()
