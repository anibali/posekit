import json

import numpy as np
import pytest
import torch
from glupy.math import ensure_homogeneous
from numpy.testing import assert_allclose
from posekit.camera import CameraIntrinsics
from posekit.normaliser import normalise_points, denormalise_points, infer_z_depth
from posekit.skeleton import skeleton_registry


class TestNormaliser:
    @pytest.fixture
    def camera(self, data_dir):
        with data_dir.joinpath('example01_camera.json').open('r') as f:
            camera_params = json.load(f)
        return CameraIntrinsics(np.asarray(camera_params['intrinsic'])[:3])

    @pytest.fixture
    def points(self, data_dir):
        return np.loadtxt(data_dir.joinpath('example01_univ_annot3.txt'))

    @pytest.fixture
    def skeleton(self):
        return skeleton_registry['mpi3d_28j']

    @pytest.fixture
    def z_ref(self):
        return 3992.29

    def test_normalise_points(self, points, camera, z_ref, skeleton):
        denorm_pose = ensure_homogeneous(points.copy(), d=3)
        denorm_pose[:, :2] -= denorm_pose[skeleton.root_joint_id, :2]
        norm_pose = normalise_points(denorm_pose, z_ref, camera, 2048, 2048)
        assert_allclose(norm_pose[skeleton.root_joint_id], np.asarray([0.0, 0.0, 0.0, 1.0]))
        actual = norm_pose[1]
        expected = torch.as_tensor([ 0.0215, -0.1514, -0.0127,  1.0000])
        assert_allclose(actual, expected, rtol=0, atol=1e-4)

    def test_denormalise_points(self, points, camera, z_ref, skeleton):
        denorm_pose = ensure_homogeneous(points.copy(), d=3)
        denorm_pose[:, :2] -= denorm_pose[skeleton.root_joint_id, :2]
        norm_pose = normalise_points(denorm_pose, z_ref, camera, 2048, 2048)
        recons_pose = denormalise_points(norm_pose, z_ref, camera, 2048, 2048)
        assert_allclose(recons_pose, denorm_pose, rtol=0, atol=1e-4)

    def test_infer_z_depth(self, points, camera, z_ref, skeleton):
        denorm_pose = ensure_homogeneous(points.copy(), d=3)
        denorm_pose[:, :2] -= denorm_pose[skeleton.root_joint_id, :2]
        norm_pose = normalise_points(denorm_pose, z_ref, camera, 2048, 2048)

        right_wrist = skeleton.joint_index('right_wrist')
        right_elbow = skeleton.joint_index('right_elbow')
        target_forearm_length = np.linalg.norm(denorm_pose[right_wrist] - denorm_pose[right_elbow])
        def eval_scale(skel):
            forearm_length = np.linalg.norm(skel[right_wrist] - skel[right_elbow])
            return forearm_length / target_forearm_length

        inferred = infer_z_depth(norm_pose, eval_scale, camera, 2048, 2048)
        assert inferred == pytest.approx(z_ref, rel=0, abs=1e-2)
