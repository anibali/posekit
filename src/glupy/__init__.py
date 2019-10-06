from time import perf_counter

import OpenGL.GL as gl
import OpenGL.GLUT as glut
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

        self._program = program
        self._prev_prog_binding = 0

    def assert_current(self):
        assert gl.glGetIntegerv(gl.GL_CURRENT_PROGRAM) == self._program

    def get_uniform_location(self, name):
        loc = gl.glGetUniformLocation(self._program, name)
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

    def __enter__(self):
        self._prev_prog_binding = gl.glGetIntegerv(gl.GL_CURRENT_PROGRAM)
        gl.glUseProgram(self._program)

    def __exit__(self, exc_type, exc_val, exc_tb):
        gl.glUseProgram(self._prev_prog_binding)

    def __del__(self):
        if gl.glDeleteProgram is not None:
            gl.glDeleteProgram(self._program)


class VAO:
    def __init__(self):
        self._vao = gl.glGenVertexArrays(1)
        self._prev_vao_binding = 0

    def assert_bound(self):
        assert gl.glGetIntegerv(gl.GL_VERTEX_ARRAY_BINDING) == self._vao

    def create_vbo(self, program: ShaderProgram, data: np.ndarray):
        self.assert_bound()
        vbo = VBO()
        stride = data.strides[0]
        for name, (dtype, offset) in data.dtype.fields.items():
            loc = gl.glGetAttribLocation(program._program, name)
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
        gl.glBindVertexArray(self._vao)

    def __exit__(self, exc_type, exc_val, exc_tb):
        gl.glBindVertexArray(self._prev_vao_binding)

    def __del__(self):
        if gl.glDeleteVertexArrays is not None:
            gl.glDeleteVertexArrays(1, [self._vao])


class VBO:
    def __init__(self):
        self._vbo = gl.glGenBuffers(1)
        self._prev_vbo_binding = 0

    def assert_bound(self):
        assert gl.glGetIntegerv(gl.GL_ARRAY_BUFFER_BINDING) == self._vbo

    def transfer_data_to_gpu(self, data: np.ndarray):
        self.assert_bound()
        gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, gl.GL_DYNAMIC_DRAW)

    def bind(self):
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo)

    def __enter__(self):
        self._prev_vbo_binding = gl.glGetIntegerv(gl.GL_ARRAY_BUFFER_BINDING)
        self.bind()

    def __exit__(self, exc_type, exc_val, exc_tb):
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._prev_vbo_binding)

    def __del__(self):
        if gl.glDeleteBuffers is not None:
            gl.glDeleteBuffers(1, [self._vbo])


class EBO:
    def __init__(self):
        self._ebo = gl.glGenBuffers(1)
        self._prev_ebo_binding = 0

    def assert_bound(self):
        assert gl.glGetIntegerv(gl.GL_ELEMENT_ARRAY_BUFFER_BINDING) == self._ebo

    def transfer_data_to_gpu(self, data: np.ndarray):
        self.assert_bound()
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, data.nbytes, data, gl.GL_DYNAMIC_DRAW)

    def bind(self):
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self._ebo)

    def __enter__(self):
        self._prev_ebo_binding = gl.glGetIntegerv(gl.GL_ELEMENT_ARRAY_BUFFER_BINDING)
        self.bind()

    def __exit__(self, exc_type, exc_val, exc_tb):
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self._prev_ebo_binding)

    def __del__(self):
        if gl.glDeleteBuffers is not None:
            gl.glDeleteBuffers(1, [self._ebo])


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


import enum
class Key(enum.Enum):
    F1 = glut.GLUT_KEY_F1
    F2 = glut.GLUT_KEY_F2
    F3 = glut.GLUT_KEY_F3
    F4 = glut.GLUT_KEY_F4
    F5 = glut.GLUT_KEY_F5
    F6 = glut.GLUT_KEY_F6
    F7 = glut.GLUT_KEY_F7
    F8 = glut.GLUT_KEY_F8
    F9 = glut.GLUT_KEY_F9
    F10 = glut.GLUT_KEY_F10
    F11 = glut.GLUT_KEY_F11
    F12 = glut.GLUT_KEY_F12
    LEFT = glut.GLUT_KEY_LEFT
    UP = glut.GLUT_KEY_UP
    RIGHT = glut.GLUT_KEY_RIGHT
    DOWN = glut.GLUT_KEY_DOWN
    PAGE_UP = glut.GLUT_KEY_PAGE_UP
    PAGE_DOWN = glut.GLUT_KEY_PAGE_DOWN
    HOME = glut.GLUT_KEY_HOME
    END = glut.GLUT_KEY_END
    INSERT = glut.GLUT_KEY_INSERT


class Keyboard:
    def __init__(self):
        self._down_keys = set()

    def fire_key_down(self, key):
        self._down_keys.add(key)

    def fire_key_up(self, key):
        self._down_keys.remove(key)

    def is_down(self, key):
        if isinstance(key, str):
            key = key.encode()
        if isinstance(key, Key):
            key = key.value
        return key in self._down_keys


class OpenGlApp:
    def __init__(self, title, width, height):
        glut.glutInit()
        glut.glutInitContextVersion(3, 3)
        glut.glutInitContextProfile(glut.GLUT_CORE_PROFILE)
        glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGBA)
        glut.glutCreateWindow(title)
        glut.glutReshapeWindow(width, height)
        glut.glutReshapeFunc(self._reshape)
        glut.glutDisplayFunc(self._display)
        glut.glutCloseFunc(self._close)
        glut.glutIdleFunc(self._idle)

        self.keyboard = Keyboard()
        glut.glutSetKeyRepeat(glut.GLUT_KEY_REPEAT_OFF)
        def on_key_down(key, x, y):
            if key == b'\x1b':  # Escape key
                glut.glutLeaveMainLoop()
            else:
                self.keyboard.fire_key_down(key)
        def on_key_up(key, x, y):
            self.keyboard.fire_key_up(key)
        glut.glutKeyboardFunc(on_key_down)
        glut.glutSpecialFunc(on_key_down)
        glut.glutKeyboardUpFunc(on_key_up)
        glut.glutSpecialUpFunc(on_key_up)

        self.last_time = perf_counter()

    def _idle(self):
        glut.glutPostRedisplay()

    def _display(self):
        cur_time = perf_counter()
        dt = cur_time - self.last_time
        self.last_time = cur_time
        self.update(dt)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        self.render(dt)
        glut.glutSwapBuffers()

    def _reshape(self, width, height):
        self.on_reshape(width, height)
        gl.glViewport(0, 0, width, height)

    def _close(self):
        self.on_close()

    def on_reshape(self, width, height):
        pass

    def on_close(self):
        pass

    def update(self, dt):
        pass

    def render(self, dt):
        pass

    def run(self):
        glut.glutMainLoop()
