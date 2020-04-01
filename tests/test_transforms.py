import json

import PIL.Image
import numpy as np
import pytest
from posekit.transforms.transformers import Tradeoff

from glupy.math import to_cartesian
from numpy.testing import assert_allclose
from posekit import transforms
from posekit.camera import CameraIntrinsics
from posekit.skeleton import skeleton_registry
from posekit.skeleton.utils import is_pose_similar
from posekit.transforms import TransformerContext


def assert_synced(camera_a, image_a, points_a, camera_b, image_b, points_b):
    """Assert that the results of a transformation project into image space correctly.

    The 3D points are projected into 2D space, and the corresponding pixels are sampled from
    the image. These are then compared to values obtained the same way from before the
    transformation to determine whether they agree.
    """
    points2d_a = camera_a.project_cartesian(points_a)
    points2d_b = camera_b.project_cartesian(points_b)

    expected = [image_b.getpixel((x, y)) for x, y in points2d_b.tolist()]
    actual = [image_a.getpixel((x, y)) for x, y in points2d_a.tolist()]
    assert_allclose(actual, expected, rtol=0, atol=8.0)


def assert_transform_equality(ctx):
    camera = ctx.orig_camera.clone()
    camera.x_0, camera.y_0 = 0, 0

    lhs = ctx.image_transformer.matrix @ camera.matrix
    rhs = (ctx.camera_transformer.matrix @ camera.matrix) @ ctx.point_transformer.matrix

    assert_allclose(lhs, rhs, rtol=0, atol=1e-6)


def assert_untransform_consistency(ctx, camera, image, points):
    rcamera, rimage, rpoints = ctx.untransform(*ctx.transform(camera, image, points))
    assert_allclose(to_cartesian(rpoints), points)
    assert_allclose(rcamera.matrix, camera.matrix)


class TestTransforms:
    @pytest.fixture(params=['example01', 'example02'])
    def example(self, request):
        return request.param

    @pytest.fixture
    def image(self, data_dir, example):
        return PIL.Image.open(data_dir.joinpath(f'{example}_image.jpg'))

    @pytest.fixture
    def camera(self, data_dir, example):
        with data_dir.joinpath(f'{example}_camera.json').open('r') as f:
            camera_params = json.load(f)
        return CameraIntrinsics(np.asarray(camera_params['intrinsic'])[:3])

    @pytest.fixture
    def points(self, data_dir, example):
        return np.loadtxt(data_dir.joinpath(f'{example}_univ_annot3.txt'))

    @pytest.fixture
    def skeleton(self, example):
        if example == 'example01':
            return skeleton_registry['mpi3d_28j']
        elif example == 'example02':
            return skeleton_registry['mpi3d_17j']
        raise ValueError()

    def test_pan(self, camera, image, points):
        ctx = TransformerContext(camera, image.width, image.height, msaa=1)
        ctx.add(transforms.PanImage(50, -20))

        transformed = ctx.transform(camera, image, points)
        assert_synced(*transformed, camera, image, points)
        assert_transform_equality(ctx)
        assert_untransform_consistency(ctx, camera, image, points)

    def test_pan_desynced(self, camera, image, points):
        ctx = TransformerContext(camera, image.width, image.height, msaa=1,
                                 tradeoff=Tradeoff.DESYNC_IMAGE_SPACE)
        ctx.add(transforms.PanImage(50, -20))

        transformed = ctx.transform(camera, image, points)
        assert is_pose_similar(transformed[2], points, tolerance=1e-2)
        assert_untransform_consistency(ctx, camera, image, points)

    def test_zoom(self, camera, image, points):
        ctx = TransformerContext(camera, image.width, image.height, msaa=1)
        ctx.add(transforms.ZoomImage(1.4))

        transformed = ctx.transform(camera, image, points)
        assert_synced(*transformed, camera, image, points)
        assert_transform_equality(ctx)
        assert_untransform_consistency(ctx, camera, image, points)

    def test_hflip(self, camera, image, points, skeleton):
        ctx = TransformerContext(camera, image.width, image.height, msaa=1)
        ctx.add(transforms.HorizontalFlip(skeleton.hflip_indices, True))

        transformed = ctx.transform(camera, image, points)
        assert_synced(*transformed, camera, image, points[skeleton.hflip_indices])
        assert_transform_equality(ctx)
        assert_untransform_consistency(ctx, camera, image, points)

    def test_rotate(self, camera, image, points):
        ctx = TransformerContext(camera, image.width, image.height, msaa=1)
        ctx.add(transforms.RotateImage(30))

        transformed = ctx.transform(camera, image, points)
        assert_synced(*transformed, camera, image, points)
        assert_transform_equality(ctx)
        assert_untransform_consistency(ctx, camera, image, points)

    def test_rotate_desynced(self, camera, image, points):
        ctx = TransformerContext(camera, image.width, image.height, msaa=1,
                                 tradeoff=Tradeoff.DESYNC_IMAGE_SPACE)
        ctx.add(transforms.RotateImage(30))

        transformed = ctx.transform(camera, image, points)
        assert is_pose_similar(transformed[2], points, tolerance=1e-2)
        assert_untransform_consistency(ctx, camera, image, points)

    def test_square_crop(self, camera, image, points, example):
        if example == 'example01':
            pytest.skip()

        ctx = TransformerContext(camera, image.width, image.height, msaa=1)
        ctx.add(transforms.SquareCrop())
        ctx.add(transforms.ChangeResolution(256, 256))

        transformed = ctx.transform(camera, image, points)
        assert transformed[1].size == (256, 256)
        assert_synced(*transformed, camera, image, points)
        assert_transform_equality(ctx)
        assert_untransform_consistency(ctx, camera, image, points)
