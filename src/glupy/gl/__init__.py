import enum
import re
import weakref
from abc import abstractmethod, ABC
from contextlib import contextmanager
from time import perf_counter
from typing import Optional, List, Callable, TypeVar, Generic

import OpenGL.GL as gl
import glfw
import numpy as np


def np_to_gl_type(np_type: np.dtype):
    np_type = np_type.base
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


T = TypeVar('T')
class WeakStack(ABC, Generic[T]):
    def __init__(self, callback: Callable[[Optional[T], Optional[T]], None]):
        """Create a stack containing weak references to values.

        Items in the stack will be replaced with None when there are no other references to them.

        Args:
            callback: Function to be called when the top of the stack changes. It will be passed
                the old top value and new top value as arguments.
        """
        self._list: List[weakref.ReferenceType] = []
        self._callback = callback

    def empty(self):
        """Return True if and only if there are no items on the stack.
        """
        return len(self._list) == 0

    def peek(self) -> Optional[T]:
        """Return the top item of the stack.

        Returns:
            The top item of the stack if the stack is not empty, otherwise None.
        """
        if self.empty():
            return None
        return self._list[-1]()

    def _push(self, value: T):
        self._list.append(weakref.ref(value))

    def _pop(self) -> T:
        return self._list.pop()()

    def set_top(self, value: T):
        """Replace the top of the stack with a new value.

        Args:
            value: The new value for the top of the stack.
        """
        if self.empty():
            old_top = None
        else:
            old_top = self._pop()
        self._push(value)
        self._callback(old_top, value)

    def push(self, value: T):
        """Push a new item onto the stack.

        Args:
            value: The new item.
        """
        prev_top = self.peek()
        self._push(value)
        self._callback(prev_top, value)

    def pop(self) -> T:
        """Remove the item on top of the stack.

        Returns:
            The removed item.
        """
        old_top = self._pop()
        top = self.peek()
        self._callback(old_top, top)
        return old_top


def _rebind_context(old, new):
    if new is None or new is old:
        return
    glfw.make_context_current(new.window)


class Context:
    stack: WeakStack['Context'] = WeakStack(_rebind_context)

    def __init__(self, window):
        assert window is not None
        self.window = window
        self.gl_objects = weakref.WeakSet()

    @classmethod
    def get_current(cls) -> Optional['Context']:
        return cls.stack.peek()

    def make_current(self):
        Context.stack.set_top(self)

    def __enter__(self):
        Context.stack.push(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        Context.stack.pop()

    def try_destroy_gl_object(self, obj):
        if self.window is None:
            # Can't destroy the object if its context has already been destroyed.
            return False
        # If an object gets garbage collected while another context is current, temporarily make
        # this context current and destroy the object.
        with self:
            obj.destroy()
        return True

    def is_destroyed(self):
        return self.window is None

    def destroy(self):
        if self.is_destroyed():
            return
        assert self is Context.get_current()
        for obj in self.gl_objects:
            obj.destroy()
        glfw.destroy_window(self.window)
        self.window = None


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

    def bind(self):
        self.bind_manager.set_top(self)

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


class VAO(_BindableGlObject):
    def __init__(self, vbo=None, ebo=None):
        super().__init__(gl.glGenVertexArrays(1))
        self._bound_vbo = None
        self._bound_ebo = None
        with self:
            if vbo is not None:
                self.bind_vbo(vbo)
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
        vbo.bind()
        self._bound_vbo = vbo

    def bind_ebo(self, ebo):
        assert self.is_currently_bound()
        ebo.bind()
        self._bound_ebo = ebo

    def draw_elements(self, mode=gl.GL_TRIANGLES):
        assert self.is_currently_bound()
        assert self.ebo is not None
        self.ebo.draw_elements(mode)

    def _destroy(self):
        if gl.glDeleteVertexArrays is not None:
            gl.glDeleteVertexArrays(1, [self._handle])


class VBO(_BindableGlObject):
    def __init__(self, program=None, dtype=None):
        super().__init__(gl.glGenBuffers(1))
        if program is not None:
            assert dtype is not None
            self.parse_vertex_attributes(program, dtype)

    @classmethod
    def _bind(cls, handle):
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, handle)

    @classmethod
    def _get_currently_bound(cls):
        return gl.glGetIntegerv(gl.GL_ARRAY_BUFFER_BINDING)

    def parse_vertex_attributes(self, program: ShaderProgram, dtype: np.dtype):
        vertex_attributes = []
        stride = dtype.itemsize
        for name, (sub_dtype, offset) in dtype.fields.items():
            loc = program.get_attribute_location(name)
            gl_type = np_to_gl_type(sub_dtype)
            vertex_attributes.append((loc, sub_dtype.shape[0], gl_type, False, stride,
                                     gl.ctypes.c_void_p(offset)))
        self.vertex_attributes = vertex_attributes

    def connect_vertex_attributes(self):
        assert self.is_currently_bound()
        for args in self.vertex_attributes:
            loc = args[0]
            gl.glEnableVertexAttribArray(loc)
            gl.glVertexAttribPointer(*args)

    def transfer_data_to_gpu(self, data: np.ndarray):
        assert self.is_currently_bound()
        gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, gl.GL_DYNAMIC_DRAW)

    def _destroy(self):
        if gl.glDeleteBuffers is not None:
            gl.glDeleteBuffers(1, [self._handle])


class EBO(_BindableGlObject):
    def __init__(self, *, usage=gl.GL_DYNAMIC_DRAW):
        super().__init__(gl.glGenBuffers(1))
        self.usage = usage
        self.length = 0
        self.dtype = 0

    @classmethod
    def _bind(cls, handle):
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, handle)

    @classmethod
    def _get_currently_bound(cls):
        return gl.glGetIntegerv(gl.GL_ELEMENT_ARRAY_BUFFER_BINDING)

    def transfer_data_to_gpu(self, data: np.ndarray):
        assert self.is_currently_bound()
        self.length = len(data)
        self.dtype = np_to_gl_type(data.dtype)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, data.nbytes, data, self.usage)

    def draw_elements(self, mode):
        gl.glDrawElements(mode, self.length, self.dtype, None)

    def _destroy(self):
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
    ENTER = glfw.KEY_ENTER


class ModifierKey(enum.Enum):
    SHIFT = glfw.MOD_SHIFT
    CONTROL = glfw.MOD_CONTROL
    ALT = glfw.MOD_ALT
    SUPER = glfw.MOD_SUPER
    CAPS_LOCK = glfw.MOD_CAPS_LOCK
    NUM_LOCK = glfw.MOD_NUM_LOCK


_modifier_map = {
    'alt': ModifierKey.ALT,
    'ctrl': ModifierKey.CONTROL,
    'shift': ModifierKey.SHIFT,
    'super': ModifierKey.SUPER,
}


def _clean_modifier_key(mod_key):
    if isinstance(mod_key, str):
        mod_key = mod_key.upper()
        if mod_key == 'CTRL':
            mod_key = ModifierKey.CONTROL
        else:
            mod_key = ModifierKey[mod_key]
    return mod_key


def _clean_key(key, mod_keys):
    if isinstance(key, KeyboardShortcut):
        key, mod_keys = key.base_key, key.modifier_keys
    if isinstance(key, str):
        if len(key) == 1:
            key = ord(key.upper())
        else:
            key = Key[key.upper()]
    if isinstance(key, Key):
        key = key.value
    if mod_keys is not None:
        mod_keys = [_clean_modifier_key(mod_key) for mod_key in mod_keys]
    return key, mod_keys


class KeyboardShortcut:
    _parser_regex = re.compile(r'((?:\w+\+)*)(.*)')

    def __init__(self, base_key, modifier_keys=None):
        if modifier_keys is None:
            modifier_keys = []
        self.base_key = base_key
        self.modifier_keys = set(modifier_keys)

    @classmethod
    def parse(cls, string_shortcut):
        match = cls._parser_regex.match(string_shortcut)
        if not match or len(match.groups()) != 2:
            raise ValueError(f'malformed shortcut: {string_shortcut}')
        modifiers, base = match.groups()
        modifiers = modifiers.lower().split('+')[:-1]
        base_key, modifier_keys = _clean_key(base, modifiers)
        return cls(base_key, modifier_keys)


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

    def is_down(self, key, mod_keys=None):
        key, mod_keys = _clean_key(key, mod_keys)
        return key in self._down_keys and self.has_exact_modifiers(mod_keys)

    def is_up(self, key, mod_keys=None):
        return not self.is_down(key, mod_keys)

    def was_released(self, key, mod_keys=None):
        key, mod_keys = _clean_key(key, mod_keys)
        return key in self._released_keys and self.has_exact_modifiers(mod_keys)

    def was_pressed(self, key, mod_keys=None):
        key, mod_keys = _clean_key(key, mod_keys)
        return key in self._pressed_keys and self.has_exact_modifiers(mod_keys)

    def has_modifier(self, mod_key: ModifierKey):
        return (self._modifiers & mod_key.value) != 0

    def has_exact_modifiers(self, mod_keys):
        if mod_keys is None:
            return True
        cur_modifiers = 0
        for mod_key in mod_keys:
            cur_modifiers |= mod_key.value
        mask = ModifierKey.ALT.value | ModifierKey.CONTROL.value | ModifierKey.SHIFT.value | ModifierKey.SUPER.value
        return self._modifiers & mask == cur_modifiers & mask

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
        self.context = Context(self.window)
        self.context.make_current()

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
