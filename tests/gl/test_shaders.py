import numpy as np

from glupy.math import mat4
from posekit.gui.shaders import create_image_shader, create_seekbar_shader, \
    create_checkerboard_shader, create_simple_shader


def test_image_shader(gl_context):
    shader = create_image_shader()
    with shader:
        shader.set_uniform_mat4('transProj', mat4.identity())


def test_seekbar_shader(gl_context):
    shader = create_seekbar_shader()
    with shader:
        shader.set_uniform_mat4('transProj', mat4.identity())
        shader.set_uniform_float('progress', 0.5)


def test_checkerboard_shader(gl_context):
    shader = create_checkerboard_shader()
    with shader:
        shader.set_uniform_mat4('modelMatrix', mat4.identity())


def test_simple_shader(gl_context):
    shader = create_simple_shader()
    with shader:
        shader.set_uniform_mat4('modelMatrix', mat4.identity())
        shader.set_uniform_vec4('color', np.asarray([0.0, 0.0, 0.0, 0.0]))
