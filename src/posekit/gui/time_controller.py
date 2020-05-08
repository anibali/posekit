import enum

import numpy as np

from glupy.gl import KeyboardShortcut


class TimeController:
    @enum.unique
    class Action(enum.Enum):
        PLAY_PAUSE = enum.auto()
        SPEED_UP = enum.auto()
        SPEED_DOWN = enum.auto()
        NEXT_FRAME = enum.auto()
        PREVIOUS_FRAME = enum.auto()
        FORWARD_10_FRAMES = enum.auto()
        BACKWARD_10_FRAMES = enum.auto()
        FIRST_FRAME = enum.auto()
        LAST_FRAME = enum.auto()

    def __init__(self, total_frames, frame_rate, do_seek_frame, do_next_frame):
        self.final_time = (total_frames - 1) / frame_rate
        self.frame_rate = frame_rate
        self.do_seek_frame = do_seek_frame
        self.do_next_frame = do_next_frame

        self.cur_time = 0
        self.prev_frame = 0
        self.next_frame = 0

        self.paused = False
        self.looping = True
        self.playback_speed = 1
        self.absolute_time = None

        self.set_default_keymap()

    def set_keymap(self, **kwargs):
        self.keymap = {}
        for action_name, shortcut in kwargs.items():
            action = self.Action[action_name.upper()]
            self.keymap[action] = KeyboardShortcut.parse(shortcut)

    def set_default_keymap(self):
        self.set_keymap(
            play_pause=' ',
            speed_up=']',
            speed_down='[',
            next_frame='.',
            previous_frame=',',
            forward_10_frames='shift+.',
            backward_10_frames='shift+,',
            first_frame=';',
            last_frame='\'',
        )

    @property
    def current_frame(self):
        return int(self.cur_time * self.frame_rate)

    @property
    def progress(self):
        return self.cur_time / self.final_time

    @progress.setter
    def progress(self, value):
        value = np.clip(value, 0.0, 1.0)
        self.absolute_time = value * self.final_time

    def increase_playback_speed(self):
        self.playback_speed = min(self.playback_speed * 2, 2**4)

    def decrease_playback_speed(self):
        self.playback_speed = max(self.playback_speed / 2, 2**-4)

    def toggle_paused(self):
        self.paused = not self.paused
        # Reset playback speed to 1x speed forward.
        self.playback_speed = 1

    def step_frames(self, n):
        self.cur_time += n / self.frame_rate

    def fire_seek_frame(self, frame):
        frame = int(frame)
        self.do_seek_frame(frame)
        self.next_frame = frame

    def fire_next_frame(self):
        self.do_next_frame(self.next_frame)
        self.prev_frame = self.next_frame
        self.next_frame += 1

    def handle_keyboard_input(self, keyboard):
        for action, shortcut in self.keymap.items():
            if keyboard.was_pressed(shortcut):
                break
        else:
            action = None

        if action == self.Action.PLAY_PAUSE:
            self.toggle_paused()
        if action == self.Action.SPEED_UP:
            self.increase_playback_speed()
        if action == self.Action.SPEED_DOWN:
            self.decrease_playback_speed()
        if action == self.Action.NEXT_FRAME:
            self.step_frames(1)
        if action == self.Action.PREVIOUS_FRAME:
            self.step_frames(-1)
        if action == self.Action.FORWARD_10_FRAMES:
            self.step_frames(10)
        if action == self.Action.BACKWARD_10_FRAMES:
            self.step_frames(-10)
        if action == self.Action.FIRST_FRAME:
            self.absolute_time = 0
        if action == self.Action.LAST_FRAME:
            self.absolute_time = self.final_time

    def update(self, dt):
        # Adjust the current time.
        if self.absolute_time is None:
            if not self.paused:
                self.cur_time += dt * self.playback_speed
        else:
            self.cur_time = self.absolute_time
        self.absolute_time = None

        # Handle time outside temporal extents.
        if self.cur_time > self.final_time:
            self.cur_time = 0 if self.looping else self.final_time
        if self.cur_time < 0:
            self.cur_time = self.final_time if self.looping else 0

        # Fire events.
        cur_frame = self.cur_time * self.frame_rate
        if cur_frame < self.prev_frame or cur_frame > self.next_frame + 5:
            self.fire_seek_frame(cur_frame)
        while self.next_frame <= cur_frame:
            self.fire_next_frame()
