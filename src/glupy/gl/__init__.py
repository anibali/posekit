import ctypes as C
import weakref
from abc import abstractmethod, ABC
from contextlib import contextmanager
from typing import Optional

import OpenGL.GL as gl
import numpy as np

from glupy.gl.app import *
from glupy.gl.context import Context
from glupy.gl.input import *
from glupy.utils.weak_stack import WeakStack


def np_to_gl_type(np_type: np.dtype):
    if np_type == np.int8:
        return gl.GL_BYTE
    if np_type == np.uint8:
        return gl.GL_UNSIGNED_BYTE
    if np_type == np.int16:
        return gl.GL_SHORT
    if np_type == np.uint16:
        return gl.GL_UNSIGNED_SHORT
    if np_type == np.int32:
        return gl.GL_INT
    if np_type == np.uint32:
        return gl.GL_UNSIGNED_INT
    if np_type == np.float16:
        return gl.GL_HALF_FLOAT
    if np_type == np.float32:
        return gl.GL_FLOAT
    if np_type == np.float64:
        return gl.GL_DOUBLE
    raise TypeError(f'Unsupported base data type: {np_type}')


class _GlObject(ABC):
    def __init__(self, handle):
        assert handle is not None
        self._handle = handle
        self._context = Context.get_current()
        assert self._context is not None
        self._context.gl_objects.add(self)

    @abstractmethod
    def _destroy(self):
        pass

    def is_destroyed(self):
        return self._handle is None

    def destroy(self):
        if self.is_destroyed():
            return
        self._destroy()
        self._handle = None

    def __del__(self):
        self._context.try_destroy_gl_object(self)


class _BindableGlObject(_GlObject):
    def __init__(self, handle):
        super().__init__(handle)
        cls = self.__class__
        if not hasattr(cls, '_bind_managers'):
            cls._bind_managers = weakref.WeakKeyDictionary()
        if self._context not in cls._bind_managers:
            cls._bind_managers[self._context] = WeakStack(cls._rebind_fn)

    @classmethod
    def _rebind_fn(cls, old, new):
        if new is old:
            return
        if new is None or new.is_destroyed():
            handle = 0
        else:
            handle = new._handle
        cls._bind(handle)

    @classmethod
    @abstractmethod
    def _bind(cls, handle):
        pass

    @classmethod
    @abstractmethod
    def _get_currently_bound(cls):
        pass

    @property
    def bind_manager(self) -> WeakStack['_BindableGlObject']:
        return self._bind_managers[self._context]

    def force_bind(self):
        self.bind_manager._set_top(self)
        self._bind(self._handle)

    def is_currently_bound(self):
        return self._get_currently_bound() == self._handle

    def __enter__(self):
        self.bind_manager.push(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.bind_manager.pop()


class ShaderProgram(_BindableGlObject):
    def __init__(self, vertex_code, fragment_code):
        super().__init__(gl.glCreateProgram())
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
        gl.glAttachShader(self._handle, vertex)
        gl.glAttachShader(self._handle, fragment)
        gl.glLinkProgram(self._handle)

        if not gl.glGetProgramiv(self._handle, gl.GL_LINK_STATUS):
            print(gl.glGetProgramInfoLog(self._handle))
            raise RuntimeError('Linking error')

        # Free shader source and unlinked object code.
        gl.glDetachShader(self._handle, vertex)
        gl.glDetachShader(self._handle, fragment)
        gl.glDeleteShader(vertex)
        gl.glDeleteShader(fragment)

    @classmethod
    def _bind(cls, handle):
        gl.glUseProgram(handle)

    @classmethod
    def _get_currently_bound(cls):
        return gl.glGetIntegerv(gl.GL_CURRENT_PROGRAM)

    def get_attribute_location(self, name):
        loc = gl.glGetAttribLocation(self._handle, name)
        if loc < 0:
            raise KeyError(f'attribute not found: {name}')
        return loc

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
        assert self.is_currently_bound()
        loc = self.get_uniform_location(name)
        gl.glUniformMatrix4fv(loc, 1, False, value)

    def set_uniform_vec4(self, name, value):
        assert self.is_currently_bound()
        loc = self.get_uniform_location(name)
        gl.glUniform4fv(loc, 1, value)

    def set_uniform_float(self, name, value):
        assert self.is_currently_bound()
        loc = self.get_uniform_location(name)
        gl.glUniform1f(loc, value)

    def _destroy(self):
        if gl.glDeleteProgram is not None:
            gl.glDeleteProgram(self._handle)


class VBO(_BindableGlObject):
    def __init__(self, data, *, usage=gl.GL_DYNAMIC_DRAW):
        super().__init__(gl.glGenBuffers(1))
        self.usage = usage
        if isinstance(data, np.dtype):
            data_layout = data
            data = None
        else:
            data_layout = np.dtype((data.dtype, data.shape))
        self.data_layout = data_layout
        with self:
            if data is None:
                self.allocate()
            else:
                self.allocate_and_write(data)

    @classmethod
    def _bind(cls, handle):
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, handle)

    @classmethod
    def _get_currently_bound(cls):
        return gl.glGetIntegerv(gl.GL_ARRAY_BUFFER_BINDING)

    def allocate(self):
        assert self.is_currently_bound()
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.data_layout.itemsize, None, self.usage)

    def write(self, data: np.ndarray):
        assert self.is_currently_bound()
        assert data.nbytes == self.data_layout.itemsize
        gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, data.nbytes, data.data)

    def allocate_and_write(self, data: np.ndarray):
        assert self.is_currently_bound()
        assert data.nbytes == self.data_layout.itemsize
        gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data.data, self.usage)

    @contextmanager
    def map(self, access=gl.GL_READ_WRITE):
        assert self.is_currently_bound()
        ptr = gl.glMapBuffer(gl.GL_ARRAY_BUFFER, access)
        size = gl.glGetBufferParameteriv(gl.GL_ARRAY_BUFFER, gl.GL_BUFFER_SIZE)
        a = np.ctypeslib.as_array(C.cast(ptr, C.POINTER(C.c_byte)), shape=(size,))
        yield a.view(self.data_layout.base)
        gl.glUnmapBuffer(gl.GL_ARRAY_BUFFER)

    def _destroy(self):
        if gl.glDeleteBuffers is not None:
            gl.glDeleteBuffers(1, [self._handle])


class EBO(_BindableGlObject):
    def __init__(self, data: np.ndarray, *, usage=gl.GL_DYNAMIC_DRAW):
        super().__init__(gl.glGenBuffers(1))
        self.usage = usage
        self.length = 0
        self.dtype = 0
        with self:
            self.allocate_and_write(data)

    @classmethod
    def _bind(cls, handle):
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, handle)

    @classmethod
    def _get_currently_bound(cls):
        return gl.glGetIntegerv(gl.GL_ELEMENT_ARRAY_BUFFER_BINDING)

    def allocate_and_write(self, data: np.ndarray):
        assert self.is_currently_bound()
        self.length = len(data)
        self.dtype = np_to_gl_type(data.dtype.base)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, data.nbytes, data.data, self.usage)

    def draw_elements(self, mode):
        gl.glDrawElements(mode, self.length, self.dtype, None)

    def _destroy(self):
        if gl.glDeleteBuffers is not None:
            gl.glDeleteBuffers(1, [self._handle])


class VAO(_BindableGlObject):
    def __init__(
        self,
        vbo: Optional[VBO] = None,
        ebo: Optional[EBO] = None,
        *,
        connect_to: Optional[ShaderProgram] = None,
    ):
        super().__init__(gl.glGenVertexArrays(1))
        self._bound_vbo = None
        self._bound_ebo = None
        with self:
            if vbo is not None:
                self.bind_vbo(vbo)
                if connect_to is not None:
                    self.connect_vertex_attributes(connect_to)
            if ebo is not None:
                self.bind_ebo(ebo)

    @property
    def ebo(self):
        return self._bound_ebo

    @property
    def vbo(self):
        return self._bound_vbo

    @classmethod
    def _bind(cls, handle):
        gl.glBindVertexArray(handle)

    @classmethod
    def _get_currently_bound(cls):
        return gl.glGetIntegerv(gl.GL_VERTEX_ARRAY_BINDING)

    def bind_vbo(self, vbo):
        assert self.is_currently_bound()
        vbo.force_bind()
        self._bound_vbo = vbo

    def bind_ebo(self, ebo):
        assert self.is_currently_bound()
        ebo.force_bind()
        self._bound_ebo = ebo

    def draw_elements(self, mode=gl.GL_TRIANGLES):
        assert self.is_currently_bound()
        assert self.ebo is not None
        self.ebo.draw_elements(mode)

    def connect_vertex_attributes(self, program: ShaderProgram):
        assert self.is_currently_bound()
        assert self._bound_vbo and self._bound_vbo.is_currently_bound()
        dtype = self._bound_vbo.data_layout.base
        for name, (sub_dtype, offset) in dtype.fields.items():
            loc = program.get_attribute_location(name)
            if self._bound_vbo.data_layout.shape == tuple():
                gl_type = np_to_gl_type(sub_dtype.base.base)
                stride = sub_dtype.base.itemsize
            else:
                gl_type = np_to_gl_type(sub_dtype.base)
                stride = dtype.itemsize
            gl.glEnableVertexAttribArray(loc)
            gl.glVertexAttribPointer(loc, sub_dtype.shape[0], gl_type, False, stride,
                                     C.c_void_p(offset))

    def _destroy(self):
        if gl.glDeleteVertexArrays is not None:
            gl.glDeleteVertexArrays(1, [self._handle])


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


class UBO(_BindableGlObject):
    def __init__(self, fields):
        super().__init__(gl.glGenBuffers(1))
        self._cpu_data = np.zeros(1, dtype=fields)

    @classmethod
    def _bind(cls, handle):
        gl.glBindBuffer(gl.GL_UNIFORM_BUFFER, handle)

    @classmethod
    def _get_currently_bound(cls):
        return gl.glGetIntegerv(gl.GL_UNIFORM_BUFFER_BINDING)

    def __setitem__(self, key, value):
        self._cpu_data[key][0] = value

    def __getitem__(self, key):
        return self._cpu_data[key][0]

    def flush(self):
        assert self.is_currently_bound()
        gl.glBufferData(gl.GL_UNIFORM_BUFFER, self._cpu_data.nbytes, self._cpu_data,
                        gl.GL_DYNAMIC_DRAW)

    def _destroy(self):
        if gl.glDeleteBuffers is not None:
            gl.glDeleteBuffers(1, [self._handle])


class Texture2d(_BindableGlObject):
    target = gl.GL_TEXTURE_2D

    def __init__(self, shape):
        super().__init__(gl.glGenTextures(1))
        h, w, c = shape

        with self:
            gl.glTexParameterf(self.target, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            gl.glTexParameterf(self.target, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
            gl.glTexParameterf(self.target, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
            gl.glTexParameterf(self.target, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
            gl.glTexParameterf(self.target, gl.GL_TEXTURE_WRAP_R, gl.GL_CLAMP_TO_EDGE)
            gl.glTexImage2D(self.target, 0, gl.GL_RGBA, w, h, 0, gl.GL_RGBA,
                            gl.GL_UNSIGNED_BYTE, None)

        self.nbytes = h * w * c
        self.shape = shape

    @classmethod
    def _bind(cls, handle):
        gl.glBindTexture(cls.target, handle)

    @classmethod
    def _get_currently_bound(cls):
        return gl.glGetIntegerv(gl.GL_TEXTURE_BINDING_2D)

    def _destroy(self):
        if gl.glDeleteTextures is not None:
            gl.glDeleteTextures(1, [self._handle])
