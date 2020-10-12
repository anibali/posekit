from hypothesis import given
from hypothesis.strategies import floats
from hypothesis.extra.numpy import arrays as ndarrays
import numpy as np
from numpy.testing import assert_array_almost_equal

from glupy.math import mat3

_sensible_floats = floats(-1e9, 1e9, allow_infinity=False)


class TestInplaceOps:
    @given(
        m=ndarrays(np.float64, (3, 3), elements=_sensible_floats),
        tx=_sensible_floats,
        ty=_sensible_floats,
    )
    def test_translate(self, m, tx, ty):
        m[2] = [0, 0, 1]
        expected = mat3.concatenate([m, mat3.translate(tx, ty)])
        actual = m.copy()
        mat3.do_translate_(actual, tx, ty)
        assert_array_almost_equal(actual, expected)

    @given(
        m=ndarrays(np.float64, (3, 3), elements=_sensible_floats),
        sx=_sensible_floats,
        sy=_sensible_floats,
    )
    def test_scale(self, m, sx, sy):
        m[2] = [0, 0, 1]
        expected = mat3.concatenate([m, mat3.scale(sx, sy)])
        actual = m.copy()
        mat3.do_scale_(actual, sx, sy)
        assert_array_almost_equal(actual, expected)

    @given(
        m=ndarrays(np.float64, (3, 3), elements=_sensible_floats),
        x=_sensible_floats,
    )
    def test_flip_x(self, m, x):
        m[2] = [0, 0, 1]
        expected = mat3.concatenate([m, mat3.flip_x(x)])
        actual = m.copy()
        mat3.do_flip_x_(actual, x)
        assert_array_almost_equal(actual, expected)
