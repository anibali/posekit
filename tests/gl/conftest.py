import glfw
import pytest

from glupy.gl import Context


@pytest.fixture
def gl_context():
    if not glfw.init():
        raise Exception('Failed to initialise GLFW.')
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(800, 600, 'Test window', None, None)
    context = Context(window)
    context.make_current()
    yield context
    context.destroy()
