import numpy as np
import pytest
from glupy.math import to_cartesian
from numpy.testing import assert_allclose
from posekit.camera import CameraIntrinsics


class TestCameraIntrinsics:
    @pytest.fixture
    def camera_intrinsics(self):
        return CameraIntrinsics.from_ccd_params(alpha_x=1200, alpha_y=1300, x_0=1000, y_0=500)

    def test_project(self, camera_intrinsics):
        coords = np.asarray([100, 200, 1000, 1], dtype=np.float64)
        actual = to_cartesian(camera_intrinsics.project(coords))
        expected = np.asarray([1120, 760], dtype=np.float64)
        assert_allclose(actual, expected, rtol=0, atol=1e-9)

    def test_back_project(self, camera_intrinsics):
        orig = np.asarray([100, 200, 1000, 1], dtype=np.float64)
        proj = np.asarray([1120, 760, 1], dtype=np.float64)
        recons = camera_intrinsics.back_project(proj)
        # Check that the original point, back-projected point, and camera centre are collinear
        rank = np.linalg.matrix_rank(
            np.stack([orig[:3], recons[:3], np.zeros(3, dtype=np.float64)])
        )
        assert rank == 1

    def test_clone(self, camera_intrinsics):
        cloned = camera_intrinsics.clone()
        assert cloned.matrix is not camera_intrinsics.matrix
        assert_allclose(cloned.matrix, camera_intrinsics.matrix)
