import weakref
from typing import Optional

import glfw

from glupy.utils.weak_stack import WeakStack


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
