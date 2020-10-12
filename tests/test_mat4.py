import numpy as np
from hypothesis import given
from hypothesis.extra.numpy import arrays as ndarrays
from hypothesis.strategies import floats
from numpy.testing import assert_array_almost_equal

from glupy.math import mat4

_sensible_floats = floats(-1e9, 1e9, allow_infinity=False)


def test_is_similarity():
    assert mat4.is_similarity(mat4.translate(4.0, 56.7, 2.3))
    assert mat4.is_similarity(mat4.rotate_axis_angle(0, 1, 0, 0.453))
    assert mat4.is_similarity(mat4.scale(1, -1, 1))

    assert not mat4.is_similarity(mat4.scale(2, 1, 1))
    assert not mat4.is_similarity(mat4.affine(A=np.random.randn(3, 3)))


class TestInplaceOps:
    @given(
        m=ndarrays(np.float64, (4, 4), elements=_sensible_floats),
        tx=_sensible_floats,
        ty=_sensible_floats,
        tz=_sensible_floats,
    )
    def test_translate(self, m, tx, ty, tz):
        m[3] = [0, 0, 0, 1]
        expected = mat4.concatenate([m, mat4.translate(tx, ty, tz)])
        actual = m.copy()
        mat4.do_translate_(actual, tx, ty, tz)
        assert_array_almost_equal(actual, expected)

    @given(
        m=ndarrays(np.float64, (4, 4), elements=_sensible_floats),
        sx=_sensible_floats,
        sy=_sensible_floats,
        sz=_sensible_floats,
    )
    def test_scale(self, m, sx, sy, sz):
        m[3] = [0, 0, 0, 1]
        expected = mat4.concatenate([m, mat4.scale(sx, sy, sz)])
        actual = m.copy()
        mat4.do_scale_(actual, sx, sy, sz)
        assert_array_almost_equal(actual, expected)

    @given(
        m=ndarrays(np.float64, (4, 4), elements=_sensible_floats),
        x=_sensible_floats,
    )
    def test_flip_x(self, m, x):
        m[3] = [0, 0, 0, 1]
        expected = mat4.concatenate([m, mat4.flip_x(x)])
        actual = m.copy()
        mat4.do_flip_x_(actual, x)
        assert_array_almost_equal(actual, expected)
