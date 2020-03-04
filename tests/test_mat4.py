import numpy as np
from glupy.math import mat4


def test_is_similarity():
    assert mat4.is_similarity(mat4.translate(4.0, 56.7, 2.3))
    assert mat4.is_similarity(mat4.rotate_axis_angle(0, 1, 0, 0.453))
    assert mat4.is_similarity(mat4.scale(1, -1, 1))

    assert not mat4.is_similarity(mat4.scale(2, 1, 1))
    assert not mat4.is_similarity(mat4.affine(A=np.random.randn(3, 3)))
