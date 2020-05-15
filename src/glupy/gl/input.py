import enum
import re

import glfw

__all__ = ['Key', 'ModifierKey', 'KeyboardShortcut', 'Keyboard', 'MouseButton', 'Mouse']


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
