import OpenGL.GL as gl
import numpy as np
import torch
from tvl import VideoLoader

from glupy.gl import OpenGlApp, Key, MouseButton, ModifierKey
from posekit.gui.components import SeekBar
from posekit.gui.torch_components import OrthImage


class VideoPlayer(OpenGlApp):
    def __init__(self, video_path, window_title='Video player', window_width=1280, window_height=720):
        super().__init__(window_title, window_width, window_height)

        # Enable alpha blending.
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        vl = VideoLoader(video_path, device='cuda', dtype=torch.uint8)
        video_height = vl.height
        video_width = vl.width

        self.vl = vl
        self.cur_time = 0
        self.next_time = 0
        self.paused = False
        self.playback_speed = 1

        # Ensure PyTorch CUDA is initialised (it is important that this happens _after_ PyCUDA
        # initialises its context, which is currently done via autoinit).
        assert torch.cuda.is_available()
        torch.empty(1, device='cuda')

        self.orth_image = OrthImage(video_width, video_height)
        self.seekbar = SeekBar()

        self.on_reshape(self.window_width, self.window_height)

    def on_reshape(self, width, height):
        self.orth_image.on_reshape(width, height)
        self.seekbar.on_reshape(width, height)

    def update(self, dt):
        if self.keyboard.was_pressed(' '):
            self.paused = not self.paused
            # Reset playback speed to 1x speed forward.
            self.playback_speed = 1
        if self.keyboard.was_pressed(Key.RIGHT):
            if self.playback_speed > 0:
                self.playback_speed = min(self.playback_speed * 2, 16)
            elif self.playback_speed < -1:
                self.playback_speed = min(self.playback_speed / 2, 1)
            else:
                self.playback_speed = 1
        if self.keyboard.was_pressed(Key.LEFT):
            if self.playback_speed < 0:
                self.playback_speed = min(self.playback_speed * 2, 16)
            elif self.playback_speed > 1:
                self.playback_speed = min(self.playback_speed / 2, 1)
            else:
                self.playback_speed = -1
        if self.keyboard.was_pressed('.'):
            if self.keyboard.has_modifier(ModifierKey.SHIFT):
                self.cur_time += 32 / self.vl.frame_rate
            else:
                self.cur_time += 1 / self.vl.frame_rate
        if self.keyboard.was_pressed(','):
            if self.keyboard.has_modifier(ModifierKey.SHIFT):
                self.cur_time -= 32 / self.vl.frame_rate
            else:
                self.cur_time -= 1 / self.vl.frame_rate
            self.next_time = -1
        if not self.paused:
            self.cur_time += dt * self.playback_speed
        if self.playback_speed < 0:
            self.next_time = -1
        vl = self.vl
        duration = vl.duration - 0.1
        if self.keyboard.was_pressed(Key.HOME):
            self.cur_time = 0
            self.next_time = -1
        if self.keyboard.was_pressed(Key.END):
            self.cur_time = duration
            self.next_time = -1
        if self.mouse.is_down(MouseButton.LEFT):
            if self.mouse.down_y > self.window_height - 40:
                seek_percent = self.mouse.x / self.window_width
                seek_percent = np.clip(seek_percent, 0.0, 1.0)
                self.cur_time = seek_percent * duration
                self.next_time = -1

        if self.cur_time > duration:
            self.cur_time = 0
            self.next_time = -1
        if self.cur_time < 0:
            self.cur_time = duration
            self.next_time = -1
        if self.cur_time - self.next_time > 0.25:
            vl.seek(self.cur_time)
            self.orth_image.set_image(vl.read_frame())
            self.next_time = self.cur_time + 1 / vl.frame_rate
        else:
            while self.cur_time >= self.next_time:
                self.orth_image.set_image(vl.read_frame())
                self.next_time += 1 / vl.frame_rate

        self.seekbar.set_progress(self.cur_time / duration)

    def render(self, dt):
        self.orth_image.render(dt)
        self.seekbar.render(dt)
