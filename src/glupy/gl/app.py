from time import perf_counter

import OpenGL.GL as gl
import glfw

from glupy.gl.context import Context
from glupy.gl.input import Keyboard, Mouse

__all__ = ['OpenGlApp']


class OpenGlApp:
    def __init__(self, title, width, height, msaa=1):
        self._width = width
        self._height = height

        if not glfw.init():
            raise Exception('Failed to initialise GLFW.')

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.SAMPLES, msaa)
        self.window = glfw.create_window(width, height, title, None, None)
        self.context = Context(self.window)
        self.context.make_current()

        self.keyboard = Keyboard()
        self.mouse = Mouse()

        glfw.set_window_size_callback(self.window, self._reshape)
        glfw.set_window_close_callback(self.window, self._close)
        glfw.set_key_callback(self.window, self._key)
        glfw.set_mouse_button_callback(self.window, self._mouse_button)
        glfw.set_cursor_pos_callback(self.window, self._cursor_pos)

    @property
    def window_width(self):
        return self._width

    @property
    def window_height(self):
        return self._height

    def set_title(self, title):
        glfw.set_window_title(self.window, str(title))

    def _reshape(self, window, width, height):
        self._width = width
        self._height = height
        self.on_reshape(width, height)
        gl.glViewport(0, 0, width, height)

    def _close(self, window):
        self.on_close()

    def _key(self, window, key, scancode, action, mods):
        self.keyboard.set_modifiers(mods)
        if action == glfw.PRESS:
            self.keyboard.fire_key_down(key)
        if action == glfw.RELEASE:
            self.keyboard.fire_key_up(key)
        if key == glfw.KEY_ESCAPE:
            self.quit()

    def _mouse_button(self, window, button, action, mods):
        if action == glfw.PRESS:
            self.mouse.fire_button_down(button, *glfw.get_cursor_pos(self.window))
        if action == glfw.RELEASE:
            self.mouse.fire_button_up(button, *glfw.get_cursor_pos(self.window))

    def _cursor_pos(self, window, x, y):
        self.mouse.fire_move(x, y)

    def on_reshape(self, width, height):
        pass

    def on_close(self):
        pass

    def update(self, dt):
        pass

    def render(self, dt):
        pass

    def clean_up(self):
        pass

    def quit(self):
        glfw.set_window_should_close(self.window, True)
        self._close(self.window)

    def run(self):
        prev_time = perf_counter()
        while not glfw.window_should_close(self.window):
            cur_time = perf_counter()
            dt = cur_time - prev_time
            prev_time = cur_time
            self.keyboard.update(dt)
            self.mouse.update(dt)
            self.update(dt)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            self.render(dt)
            glfw.swap_buffers(self.window)
            glfw.poll_events()
        self.clean_up()
        self.context.destroy()
