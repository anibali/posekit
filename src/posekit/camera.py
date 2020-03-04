import numpy as np
from glupy.math import to_cartesian, ensure_homogeneous
from posekit.utils import cast_array


class CameraIntrinsics:
    """Represents camera calibration intrinsics."""

    def __init__(self, matrix):
        matrix = cast_array(matrix, np.float64)
        assert matrix.shape == (3, 4), 'intrinsic matrix must be 3x4'
        self.matrix = matrix

    @classmethod
    def from_ccd_params(cls, alpha_x, alpha_y, x_0, y_0):
        """Create a CameraIntrinsics instance from CCD parameters (4 DOF)."""
        matrix = cast_array([
            [alpha_x,     0.0, x_0, 0.0],
            [    0.0, alpha_y, y_0, 0.0],
            [    0.0,     0.0, 1.0, 0.0],
        ], np.float64)
        return cls(matrix)

    def clone(self):
        return self.__class__(self.matrix.copy())

    @property
    def x_0(self):
        """Get the principle point x-coordinate (in pixels along x-axis)."""
        return float(self.matrix[0, 2])

    @x_0.setter
    def x_0(self, value):
        """Set the principle point x-coordinate (in pixels along x-axis)."""
        self.matrix[0, 2] = value

    @property
    def y_0(self):
        """Get the principle point y-coordinate (in pixels along y-axis)."""
        return float(self.matrix[1, 2])

    @y_0.setter
    def y_0(self, value):
        """Set the principle point y-coordinate (in pixels along y-axis)."""
        self.matrix[1, 2] = value

    @property
    def alpha_x(self):
        """Get the focal length (in pixels along x-axis)."""
        return float(self.matrix[0, 0])

    @alpha_x.setter
    def alpha_x(self, value):
        """Set the focal length (in pixels along x-axis)."""
        self.matrix[0, 0] = value

    @property
    def alpha_y(self):
        """Get the focal length (in pixels along y-axis)."""
        return float(self.matrix[1, 1])

    @alpha_y.setter
    def alpha_y(self, value):
        """Set the focal length (in pixels along y-axis)."""
        self.matrix[1, 1] = value

    @property
    def aspect_ratio(self):
        """Get the pixel aspect ratio."""
        return self.alpha_y / self.alpha_x

    def zoom(self, factor):
        """Zoom by adjusting the camera's focal length."""
        self.alpha_x *= factor
        self.alpha_y *= factor

    def scale_image(self, sx, sy):
        """Scale the image size."""
        self.matrix[0] *= sx
        self.matrix[1] *= sy

    def project(self, coords):
        """Project points from camera space to image space."""
        assert coords.shape[-1] == 4, 'expected homogeneous coordinates in 3D space'
        return coords @ cast_array(self.matrix.T, coords)

    def project_cartesian(self, coords):
        coords = ensure_homogeneous(coords, d=3)
        return to_cartesian(self.project(coords))

    def back_project(self, coords):
        """Project points from image space to camera space (ideal points)."""
        assert coords.shape[-1] == 3, 'expected homogeneous coordinates in 2D space'
        return coords @ cast_array(np.linalg.pinv(self.matrix).T, coords)
