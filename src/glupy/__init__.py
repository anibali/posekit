import enum
from time import perf_counter

import OpenGL.GL as gl
import glfw
import numpy as np


class ShaderProgram:
    def __init__(self, vertex_code, fragment_code):
        program = gl.glCreateProgram()
        vertex = gl.glCreateShader(gl.GL_VERTEX_SHADER)
        fragment = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)

        gl.glShaderSource(vertex, vertex_code)
        gl.glShaderSource(fragment, fragment_code)

        gl.glCompileShader(vertex)
        if not gl.glGetShaderiv(vertex, gl.GL_COMPILE_STATUS):
            error = gl.glGetShaderInfoLog(vertex).decode()
            print(error)
            raise RuntimeError('Vertex shader compilation error')

        gl.glCompileShader(fragment)
        if not gl.glGetShaderiv(fragment, gl.GL_COMPILE_STATUS):
            error = gl.glGetShaderInfoLog(fragment).decode()
            print(error)
            raise RuntimeError('Fragment shader compilation error')

        # Link individual shaders to create a shader program.
        gl.glAttachShader(program, vertex)
        gl.glAttachShader(program, fragment)
        gl.glLinkProgram(program)

        if not gl.glGetProgramiv(program, gl.GL_LINK_STATUS):
            print(gl.glGetProgramInfoLog(program))
            raise RuntimeError('Linking error')

        # Free shader source and unlinked object code.
        gl.glDetachShader(program, vertex)
        gl.glDetachShader(program, fragment)
        gl.glDeleteShader(vertex)
        gl.glDeleteShader(fragment)

        self._handle = program
        self._prev_prog_binding = 0

    def assert_current(self):
        assert gl.glGetIntegerv(gl.GL_CURRENT_PROGRAM) == self._handle

    def get_uniform_block_index(self, name):
        index = gl.glGetUniformBlockIndex(self._handle, name)
        if index < 0:
            raise KeyError(f'uniform block not found: {name}')
        return index

    def get_uniform_location(self, name):
        loc = gl.glGetUniformLocation(self._handle, name)
        if loc < 0:
            raise KeyError(f'uniform not found: {name}')
        return loc

    def set_uniform_mat4(self, name, value):
        self.assert_current()
        loc = self.get_uniform_location(name)
        gl.glUniformMatrix4fv(loc, 1, False, value)

    def set_uniform_vec4(self, name, value):
        self.assert_current()
        loc = self.get_uniform_location(name)
        gl.glUniform4fv(loc, 1, value)

    def set_uniform_float(self, name, value):
        self.assert_current()
        loc = self.get_uniform_location(name)
        gl.glUniform1f(loc, value)

    def __enter__(self):
        self._prev_prog_binding = gl.glGetIntegerv(gl.GL_CURRENT_PROGRAM)
        gl.glUseProgram(self._handle)

    def __exit__(self, exc_type, exc_val, exc_tb):
        gl.glUseProgram(self._prev_prog_binding)

    def __del__(self):
        if gl.glDeleteProgram is not None and hasattr(self, '_handle'):
            gl.glDeleteProgram(self._handle)


class VAO:
    def __init__(self):
        self._handle = gl.glGenVertexArrays(1)
        self._prev_vao_binding = 0

    def assert_bound(self):
        assert gl.glGetIntegerv(gl.GL_VERTEX_ARRAY_BINDING) == self._handle

    def create_vbo(self, program: ShaderProgram, data: np.ndarray):
        self.assert_bound()
        vbo = VBO()
        stride = data.strides[0]
        for name, (dtype, offset) in data.dtype.fields.items():
            loc = gl.glGetAttribLocation(program._handle, name)
            gl.glEnableVertexAttribArray(loc)
            if dtype.base == np.float32:
                gl_type = gl.GL_FLOAT
            else:
                raise TypeError(f'Unsupported base data type: {dtype.base}')
            with vbo:
                gl.glVertexAttribPointer(loc, dtype.shape[0], gl_type, False, stride,
                                         gl.ctypes.c_void_p(offset))
        vbo.bind()
        return vbo

    def create_ebo(self):
        self.assert_bound()
        ebo = EBO()

        ebo.bind()
        return ebo

    def __enter__(self):
        self._prev_vao_binding = gl.glGetIntegerv(gl.GL_VERTEX_ARRAY_BINDING)
        gl.glBindVertexArray(self._handle)

    def __exit__(self, exc_type, exc_val, exc_tb):
        gl.glBindVertexArray(self._prev_vao_binding)

    def __del__(self):
        if gl.glDeleteVertexArrays is not None:
            gl.glDeleteVertexArrays(1, [self._handle])


class VBO:
    def __init__(self):
        self._handle = gl.glGenBuffers(1)
        self._prev_vbo_binding = 0

    def assert_bound(self):
        assert gl.glGetIntegerv(gl.GL_ARRAY_BUFFER_BINDING) == self._handle

    def transfer_data_to_gpu(self, data: np.ndarray):
        self.assert_bound()
        gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, gl.GL_DYNAMIC_DRAW)

    def bind(self):
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._handle)

    def __enter__(self):
        self._prev_vbo_binding = gl.glGetIntegerv(gl.GL_ARRAY_BUFFER_BINDING)
        self.bind()

    def __exit__(self, exc_type, exc_val, exc_tb):
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._prev_vbo_binding)

    def __del__(self):
        if gl.glDeleteBuffers is not None:
            gl.glDeleteBuffers(1, [self._handle])


class EBO:
    def __init__(self):
        self._handle = gl.glGenBuffers(1)
        self._prev_ebo_binding = 0

    def assert_bound(self):
        assert gl.glGetIntegerv(gl.GL_ELEMENT_ARRAY_BUFFER_BINDING) == self._handle

    def transfer_data_to_gpu(self, data: np.ndarray):
        self.assert_bound()
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, data.nbytes, data, gl.GL_DYNAMIC_DRAW)

    def bind(self):
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self._handle)

    def __enter__(self):
        self._prev_ebo_binding = gl.glGetIntegerv(gl.GL_ELEMENT_ARRAY_BUFFER_BINDING)
        self.bind()

    def __exit__(self, exc_type, exc_val, exc_tb):
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self._prev_ebo_binding)

    def __del__(self):
        if gl.glDeleteBuffers is not None:
            gl.glDeleteBuffers(1, [self._handle])


class UniformBinding:
    _AUTOINC_INDEX = 0

    def __init__(self, block_name, block_fields):
        self._index = self._AUTOINC_INDEX
        self._AUTOINC_INDEX += 1
        self.block_name = block_name
        self.block_fields = block_fields

    def bind_program(self, program):
        gl.glUniformBlockBinding(program._handle, program.get_uniform_block_index(self.block_name),
                                 self._index)

    def bind_ubo(self, ubo):
        gl.glBindBufferBase(gl.GL_UNIFORM_BUFFER, self._index, ubo._handle)

    def create_ubo(self, bind=True):
        ubo = UBO(self.block_fields)
        if bind:
            self.bind_ubo(ubo)
        return ubo


class UBO:
    def __init__(self, fields):
        self._cpu_data = np.zeros(1, dtype=fields)
        self._handle = gl.glGenBuffers(1)
        self._prev_ubo_binding = 0

    def __setitem__(self, key, value):
        self._cpu_data[key][0] = value

    def __getitem__(self, key):
        return self._cpu_data[key][0]

    def assert_bound(self):
        assert gl.glGetIntegerv(gl.GL_UNIFORM_BUFFER_BINDING) == self._handle

    def flush(self):
        self.assert_bound()
        gl.glBufferData(gl.GL_UNIFORM_BUFFER, self._cpu_data.nbytes, self._cpu_data,
                        gl.GL_DYNAMIC_DRAW)

    def bind(self):
        gl.glBindBuffer(gl.GL_UNIFORM_BUFFER, self._handle)

    def __enter__(self):
        self._prev_ubo_binding = gl.glGetIntegerv(gl.GL_UNIFORM_BUFFER_BINDING)
        self.bind()

    def __exit__(self, exc_type, exc_val, exc_tb):
        gl.glBindBuffer(gl.GL_UNIFORM_BUFFER, self._prev_ubo_binding)

    def __del__(self):
        if gl.glDeleteBuffers is not None:
            gl.glDeleteBuffers(1, [self._handle])


class Texture2d:
    def __init__(self, shape):
        self.target = gl.GL_TEXTURE_2D
        self.handle = gl.glGenTextures(1)

        h, w, c = shape

        gl.glBindTexture(self.target, self.handle)
        gl.glTexParameterf(self.target, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameterf(self.target, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameterf(self.target, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameterf(self.target, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameterf(self.target, gl.GL_TEXTURE_WRAP_R, gl.GL_CLAMP_TO_EDGE)
        gl.glTexImage2D(self.target, 0, gl.GL_RGBA, w, h, 0, gl.GL_RGBA,
                        gl.GL_UNSIGNED_BYTE, None)
        gl.glBindTexture(self.target, 0)

        self.nbytes = h * w * c
        self.shape = shape

        self._prev_tex_binding = 0

    def assert_bound(self):
        assert gl.glGetIntegerv(gl.GL_TEXTURE_BINDING_2D) == self.handle

    def bind(self):
        gl.glBindTexture(self.target, self.handle)

    def __enter__(self):
        self._prev_tex_binding = gl.glGetIntegerv(gl.GL_TEXTURE_BINDING_2D)
        self.bind()

    def __exit__(self, exc_type, exc_val, exc_tb):
        gl.glBindTexture(self.target, self._prev_tex_binding)

    def __del__(self):
        if gl.glDeleteTextures is not None:
            gl.glDeleteTextures(1, [self.handle])


class Key(enum.Enum):
    F1 = glfw.KEY_F1
    F2 = glfw.KEY_F2
    F3 = glfw.KEY_F3
    F4 = glfw.KEY_F4
    F5 = glfw.KEY_F5
    F6 = glfw.KEY_F6
    F7 = glfw.KEY_F7
    F8 = glfw.KEY_F8
    F9 = glfw.KEY_F9
    F10 = glfw.KEY_F10
    F11 = glfw.KEY_F11
    F12 = glfw.KEY_F12
    LEFT = glfw.KEY_LEFT
    UP = glfw.KEY_UP
    RIGHT = glfw.KEY_RIGHT
    DOWN = glfw.KEY_DOWN
    PAGE_UP = glfw.KEY_PAGE_UP
    PAGE_DOWN = glfw.KEY_PAGE_DOWN
    HOME = glfw.KEY_HOME
    END = glfw.KEY_END
    INSERT = glfw.KEY_INSERT
    DELETE = glfw.KEY_DELETE


class ModifierKey(enum.Enum):
    SHIFT = glfw.MOD_SHIFT
    CONTROL = glfw.MOD_CONTROL
    ALT = glfw.MOD_ALT
    SUPER = glfw.MOD_SUPER
    CAPS_LOCK = glfw.MOD_CAPS_LOCK
    NUM_LOCK = glfw.MOD_NUM_LOCK


class Keyboard:
    def __init__(self):
        self._down_keys = set()
        self._released_keys = set()
        self._pressed_keys = set()
        self._released_buffer = set()
        self._pressed_buffer = set()
        self._modifiers = 0

    def fire_key_down(self, key):
        self._pressed_buffer.add(key)
        self._down_keys.add(key)

    def fire_key_up(self, key):
        if key in self._down_keys:
            self._released_buffer.add(key)
            self._down_keys.remove(key)

    def set_modifiers(self, mods):
        self._modifiers = mods

    def _clean_key(self, key):
        if isinstance(key, str):
            return ord(key)
        if isinstance(key, Key):
            return key.value
        return key

    def is_down(self, key):
        return self._clean_key(key) in self._down_keys

    def is_up(self, key):
        return not self.is_down(key)

    def was_released(self, key):
        return self._clean_key(key) in self._released_keys

    def was_pressed(self, key):
        return self._clean_key(key) in self._pressed_keys

    def has_modifier(self, mod_key: ModifierKey):
        return (self._modifiers & mod_key.value) != 0

    def update(self, dt):
        if len(self._released_buffer) > 0:
            self._released_keys = self._released_buffer
            self._released_buffer = set()
        elif len(self._released_keys) > 0:
            self._released_keys = set()
        if len(self._pressed_buffer) > 0:
            self._pressed_keys = self._pressed_buffer
            self._pressed_buffer = set()
        elif len(self._pressed_keys) > 0:
            self._pressed_keys = set()


class MouseButton(enum.Enum):
    LEFT = glfw.MOUSE_BUTTON_LEFT
    MIDDLE = glfw.MOUSE_BUTTON_MIDDLE
    RIGHT = glfw.MOUSE_BUTTON_RIGHT


class Mouse:
    def __init__(self):
        self._down_buttons = set()
        self._released_buttons = set()
        self._pressed_buttons = set()
        self._released_buffer = set()
        self._pressed_buffer = set()
        self.x = 0
        self.y = 0
        self.down_x = 0
        self.down_y = 0

    def fire_button_down(self, button, x, y):
        self._pressed_buffer.add(button)
        self._down_buttons.add(button)
        self.down_x = x
        self.down_y = y

    def fire_button_up(self, button, x, y):
        if button in self._down_buttons:
            self._released_buffer.add(button)
            self._down_buttons.remove(button)

    def fire_move(self, x, y):
        self.x = x
        self.y = y

    def _clean_button(self, button):
        if isinstance(button, MouseButton):
            return button.value
        return button

    def is_down(self, button):
        return self._clean_button(button) in self._down_buttons

    def is_up(self, button):
        return not self.is_down(button)

    def was_released(self, button):
        return self._clean_button(button) in self._released_buttons

    def was_pressed(self, button):
        return self._clean_button(button) in self._pressed_buttons

    def update(self, dt):
        if len(self._released_buffer) > 0:
            self._released_buttons = self._released_buffer
            self._released_buffer = set()
        elif len(self._released_buttons) > 0:
            self._released_buttons = set()
        if len(self._pressed_buffer) > 0:
            self._pressed_buttons = self._pressed_buffer
            self._pressed_buffer = set()
        elif len(self._pressed_buttons) > 0:
            self._pressed_buttons = set()


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
        glfw.make_context_current(self.window)

        glfw.set_window_size_callback(self.window, self._reshape)
        glfw.set_window_close_callback(self.window, self._close)
        glfw.set_key_callback(self.window, self._key)
        glfw.set_mouse_button_callback(self.window, self._mouse_button)
        glfw.set_cursor_pos_callback(self.window, self._cursor_pos)

        self.keyboard = Keyboard()
        self.mouse = Mouse()

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
            glfw.set_window_should_close(self.window, True)
            self._close(self.window)

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
