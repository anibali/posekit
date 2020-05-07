import glfw
import numpy as np
import pytest

from glupy.math import mat4
from posekit.gui.shaders import create_image_shader, create_seekbar_shader, \
    create_checkerboard_shader, create_simple_shader


@pytest.fixture
def glfw_window():
    if not glfw.init():
        raise Exception('Failed to initialise GLFW.')
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(800, 600, 'Test window', None, None)
    glfw.make_context_current(window)
    return window


def test_image_shader(glfw_window):
    shader = create_image_shader()
    with shader:
        shader.set_uniform_mat4('transProj', mat4.identity())


def test_seekbar_shader(glfw_window):
    shader = create_seekbar_shader()
    with shader:
        shader.set_uniform_mat4('transProj', mat4.identity())
        shader.set_uniform_float('progress', 0.5)


def test_checkerboard_shader(glfw_window):
    shader = create_checkerboard_shader()
    with shader:
        shader.set_uniform_mat4('modelMatrix', mat4.identity())


def test_simple_shader(glfw_window):
    shader = create_simple_shader()
    with shader:
        shader.set_uniform_mat4('modelMatrix', mat4.identity())
        shader.set_uniform_vec4('color', np.asarray([0.0, 0.0, 0.0, 0.0]))
