import OpenGL.GL as gl
import torch
from tvl import VideoLoader

from glupy.gl import OpenGlApp, MouseButton
from posekit.gui.components import SeekBar
from posekit.gui.time_controller import TimeController
from posekit.gui.torch_components import OrthImage


class VideoPlayer(OpenGlApp):
    def __init__(self, video_path, window_title='Video player', window_width=1280, window_height=720):
        super().__init__(window_title, window_width, window_height)

        # Enable alpha blending.
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        self.vl = VideoLoader(video_path, device='cuda', dtype=torch.uint8)

        self.controller = TimeController(self.vl.n_frames, self.vl.frame_rate,
                                         self.do_seek_frame, self.do_next_frame)

        # Ensure PyTorch CUDA is initialised (it is important that this happens _after_ PyCUDA
        # initialises its context, which is currently done via autoinit).
        assert torch.cuda.is_available()
        torch.empty(1, device='cuda')

        self.orth_image = OrthImage(self.vl.width, self.vl.height)
        self.seekbar = SeekBar()

        self.on_reshape(self.window_width, self.window_height)

    def on_reshape(self, width, height):
        self.orth_image.on_reshape(width, height)
        self.seekbar.on_reshape(width, height)

    def do_seek_frame(self, frame):
        self.vl.seek_to_frame(frame)

    def do_next_frame(self, frame):
        try:
            self.orth_image.set_image(self.vl.read_frame())
        except EOFError:
            self.orth_image.set_image(torch.zeros_like(self.orth_image.image))

    def clean_up(self):
        super().clean_up()
        self.orth_image = None

    def update(self, dt):
        self.controller.handle_keyboard_input(self.keyboard)
        if self.mouse.is_down(MouseButton.LEFT):
            if self.mouse.down_y > self.window_height - 40:
                self.controller.progress = self.mouse.x / self.window_width
        self.controller.update(dt)
        self.seekbar.set_progress(self.controller.progress)

    def render(self, dt):
        self.orth_image.render(dt)
        self.seekbar.render(dt)
