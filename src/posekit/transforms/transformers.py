from abc import ABC, abstractmethod
from enum import IntEnum

import numpy as np
from PIL import Image, ImageEnhance
from glupy.math import mat3, ensure_homogeneous, mat4
from posekit.camera import CameraIntrinsics
from posekit.utils import cast_array
from torchvision.transforms.functional import adjust_hue


# Not all transforms can preserve lengths in 3D space AND the correspondence between pixels and
# 2D point projections. The `Tradeoff` enum defines which trade-off is permissible when performing
# transformations.
class Tradeoff(IntEnum):
    NONE = 0  # Strictly no trade-off (not supported by all transforms)
    WARP_LENGTHS = 1  # Allow lengths in 3D space to warp (ratios between lengths are not
                      # guaranteed to remain constant)
    DESYNC_IMAGE_SPACE = 2  # Allow image-space locations to desynchronise (projections of points
                            # into image space are not guaranteed to match pixels precisely)


class Transformer(ABC):
    @abstractmethod
    def transform(self, obj):
        pass

    @abstractmethod
    def untransform(self, obj):
        pass


class MatrixBasedTransformer(Transformer):
    def __init__(self, matrix):
        super().__init__()
        self.matrix = matrix

    def _mm(self, a, b):
        a = cast_array(a, self.matrix)
        b = cast_array(b, self.matrix)
        return a @ b

    def mm(self, other):
        self.matrix = self._mm(other, self.matrix)


class CameraTransformer(MatrixBasedTransformer):
    """Camera intrinsics transformer."""

    def __init__(self):
        super().__init__(mat3.identity())
        self.sx = 1
        self.sy = 1

    def get_centred_matrix(self, camera: CameraIntrinsics):
        # Move principle point to origin
        m = mat3.translate(-camera.x_0, -camera.y_0)
        # Apply transformations
        m = self.matrix @ m
        # Restore principal point
        mat3.do_translate_(m, camera.x_0 * self.sx, camera.y_0 * self.sy)
        return m

    def zoom(self, sx, sy):
        self.mm(mat3.scale(sx, sy))
        self.sx *= sx
        self.sy *= sy

    def transform(self, camera: CameraIntrinsics):
        camera = camera.clone()
        camera.matrix = cast_array(self.get_centred_matrix(camera), camera.matrix) @ camera.matrix
        return camera

    def untransform(self, camera: CameraIntrinsics):
        camera = camera.clone()
        x_0, y_0 = camera.x_0, camera.y_0
        camera.x_0, camera.y_0 = 0, 0
        camera.matrix = cast_array(np.linalg.inv(self.matrix), camera.matrix) @ camera.matrix
        camera.x_0, camera.y_0 = x_0 / self.sx, y_0 / self.sy
        return camera


class ImageTransformer(MatrixBasedTransformer):
    """Image transformer.

    Args:
        width: Input image width
        height: Input image height
        msaa: Multisample anti-aliasing scale factor
    """

    def __init__(self, width, height, x0, y0, msaa=1):
        super().__init__(mat3.identity())
        self.msaa = msaa
        self.dest_size = np.asarray([width, height])
        self.orig_width = width
        self.orig_height = height
        self.brightness = 1
        self.contrast = 1
        self.saturation = 1
        self.hue = 0

        self.x0 = x0
        self.y0 = y0

    def adjust_colour(self, brightness, contrast, saturation, hue):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def set_output_size(self, width, height):
        self.dest_size = cast_array([width, height], self.dest_size)
        return self.dest_size

    def _transform_colour(self, image):
        enhancers = [
            (ImageEnhance.Brightness, self.brightness),
            (ImageEnhance.Contrast, self.contrast),
            (ImageEnhance.Color, self.saturation),
        ]
        for Enhancer, factor in enhancers:
            if abs(factor - 1.0) > 1e-9:
                image = Enhancer(image).enhance(factor)
        if abs(self.hue) > 1e-9:
            image = adjust_hue(image, self.hue)
        return image

    @property
    def centred_matrix(self):
        ow, oh = self.dest_size.tolist()
        matrix = mat3.concatenate([
            # Move principle point to origin
            mat3.translate(-self.x0, -self.y0),
            # Apply transformations
            self.matrix,
            # Restore principal point
            mat3.translate(self.x0 * ow / self.orig_width, self.y0 * oh / self.orig_height),
        ])
        return matrix

    def _transform_image(self, image, inverse=False):
        matrix = self.centred_matrix

        if inverse:
            matrix = np.linalg.inv(matrix)
            output_size = cast_array([self.orig_width, self.orig_height], np.int)
        else:
            output_size = self.dest_size.round().astype(np.int)

        # Scale up
        matrix = self._mm(mat3.scale(self.msaa), matrix)

        # Apply affine image transformation
        inv_matrix = np.linalg.inv(matrix)
        image = image.transform(
            tuple(output_size * self.msaa),
            Image.AFFINE,
            tuple(inv_matrix[0:2].ravel()),
            Image.BILINEAR
        )

        # Scale down to output size
        if self.msaa != 1:
            image = image.resize(tuple(output_size), Image.ANTIALIAS)

        return image

    def transform(self, image: Image.Image):
        image = self._transform_image(image, inverse=False)
        image = self._transform_colour(image)
        return image

    def untransform(self, image: Image.Image):
        return self._transform_image(image, inverse=True)


class PointTransformer(MatrixBasedTransformer):
    """3D point transformer."""

    def __init__(self):
        super().__init__(mat4.identity())
        self.shuffle_indices = []

    def is_similarity(self):
        return mat4.is_similarity(self.matrix)

    def reorder_points(self, indices):
        # Prevent shuffle indices from being set multiple times
        assert len(self.shuffle_indices) == 0
        self.shuffle_indices = indices

    def transform(self, points):
        points = ensure_homogeneous(points, d=3)
        if len(self.shuffle_indices) > 0:
            points = points[..., self.shuffle_indices, :]
        return points @ cast_array(self.matrix.T, points)

    def untransform(self, points):
        points = ensure_homogeneous(points, d=3)
        if len(self.shuffle_indices) > 0:
            inv_shuffle_indices = list(range(len(self.shuffle_indices)))
            for i, j in enumerate(self.shuffle_indices):
                inv_shuffle_indices[j] = i
            points = points[..., inv_shuffle_indices, :]
        return points @ cast_array(np.linalg.inv(self.matrix).T, points)


class TransformerContext:
    def __init__(self, camera, image_width, image_height, msaa=2,
                 tradeoff=Tradeoff.WARP_LENGTHS):
        self.orig_camera = camera
        self.camera_transformer = CameraTransformer()
        self.image_transformer = ImageTransformer(image_width, image_height, camera.x_0, camera.y_0, msaa=msaa)
        self.point_transformer = PointTransformer()
        self.tradeoff = tradeoff

    def add(self, transform, camera=True, image=True, points=True):
        if points:
            transform.add_point_transform(self)
        if camera:
            transform.add_camera_transform(self)
        if image:
            transform.add_image_transform(self)

    def transform(self, camera=None, image=None, points=None):
        pairs = [
            (camera, self.camera_transformer),
            (image, self.image_transformer),
            (points, self.point_transformer),
        ]
        return tuple([t.transform(obj) if obj is not None else None for obj, t in pairs])

    def untransform(self, camera=None, image=None, points=None):
        pairs = [
            (camera, self.camera_transformer),
            (image, self.image_transformer),
            (points, self.point_transformer),
        ]
        return tuple([t.untransform(obj) if obj is not None else None for obj, t in pairs])
