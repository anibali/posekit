import argparse
import bisect
import json
import random
from importlib import resources
from pathlib import Path

import OpenGL.GL as gl
import numpy as np
import pycuda.autoinit
import pycuda.driver
import torch
from pycuda.gl import graphics_map_flags, RegisteredImage
from tvl import VideoLoader

import glupy.examples.video_player
from glupy import VAO, ShaderProgram, OpenGlApp, Texture2d, Key, mat4, MouseButton


class MappedTexture:
    def __init__(self, height, width):
        channels = 4
        self._gl_texture = Texture2d((height, width, channels))
        self._cuda_buffer = RegisteredImage(int(self.gl_texture.handle), self.gl_texture.target,
                                            graphics_map_flags.WRITE_DISCARD)
        self._tensor = torch.zeros((height, width, channels), dtype=torch.uint8, device='cuda')

    @property
    def tensor(self):
        return self._tensor

    @property
    def gl_texture(self):
        return self._gl_texture

    def update(self):
        """Copy data from a PyTorch CUDA tensor into OpenGL texture memory."""
        tensor = self.tensor
        assert tensor.is_contiguous()
        assert tensor.numel() * tensor.element_size() == self.gl_texture.nbytes
        h, w, chans = tensor.shape
        mapping = self._cuda_buffer.map()
        memcpy = pycuda.driver.Memcpy2D()
        memcpy.set_src_device(tensor.data_ptr())
        memcpy.set_dst_array(mapping.array(0, 0))
        memcpy.height = h
        memcpy.width_in_bytes = memcpy.src_pitch = memcpy.dst_pitch = w * chans * tensor.element_size()
        memcpy(aligned=False)
        torch.cuda.synchronize(tensor.device)
        mapping.unmap()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-name', type=str, default='/aisdata/processed/ltu-hp/20191011a/20191011a-bae6')
    return parser.parse_args()


class VideoPlayer(OpenGlApp):
    def __init__(self, opts):
        super().__init__('Video player', 1280, 720)

        # Enable alpha blending.
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        base_name = opts.base_name
        videos = [
            f'{base_name}-left.mkv',
            f'{base_name}-mid.mkv',
            f'{base_name}-right.mkv',
        ]
        self.session_id, self.subject_id = Path(base_name).parts[-1].split('-')

        video_width = 1280
        video_height = 720

        vls = [
            VideoLoader(video, device='cuda', dtype=torch.uint8,
                        backend_opts={'resize': (video_height, video_width)})
            for video in videos
        ]
        self.cur_video_index = 0
        self.vls = vls
        self.image = None
        self.cur_time = 0
        self.next_time = 0
        self.paused = False
        self.playback_speed = 1

        self.aspect_ratio = video_width / video_height

        # Ensure PyTorch CUDA is initialised (it is important that this happens _after_ PyCUDA
        # initialises its context, which is currently done via autoinit).
        assert torch.cuda.is_available()
        torch.empty(1, device='cuda')

        # Create a texture buffer with OpenGL and CUDA views
        self.tex = MappedTexture(video_height, video_width)

        vertex_code = resources.read_text(glupy.examples.video_player, 'image.vert')
        fragment_code = resources.read_text(glupy.examples.video_player, 'image.frag')
        self.program = ShaderProgram(vertex_code, fragment_code)

        vertex_data = np.empty(4, [
            ('position', np.float32, 2),
            ('texcoord', np.float32, 2),
        ])

        vertex_data['position'] = [(0, 0), (0, 1), (1, 0), (1, 1)]
        vertex_data['texcoord'] = [(0, 0), (0, 1), (1, 0), (1, 1)]

        self.vao = VAO()
        with self.vao:
            vbo = self.vao.create_vbo(self.program, vertex_data)
            vbo.transfer_data_to_gpu(vertex_data)

        self.seek_program = ShaderProgram(
            resources.read_text(glupy.examples.video_player, 'image.vert'),
            resources.read_text(glupy.examples.video_player, 'seekbar.frag'),
        )

        vertex_data = np.empty(4, [
            ('position', np.float32, 2),
            ('texcoord', np.float32, 2),
        ])

        bar_h = 10  # Seek bar height (in pixels)
        vertex_data['position'] = [(0, 0), (0, bar_h), (1, 0), (1, bar_h)]
        vertex_data['texcoord'] = [(0, 0), (0, 1), (1, 0), (1, 1)]

        self.vao_seek = VAO()
        with self.vao_seek:
            vbo = self.vao_seek.create_vbo(self.seek_program, vertex_data)
            vbo.transfer_data_to_gpu(vertex_data)

        self.clip_annot_path = Path(f'{base_name}-annots.json')
        if self.clip_annot_path.is_file():
            with self.clip_annot_path.open('r') as f:
                annots = json.load(f)
            self.clips = list(sorted([
                (clip['start_time'], clip['end_time'])
                for clip in annots['clips']
            ]))
        else:
            self.clips = []
        self.annot_clip_start = None
        self.annot_clip_end = None

    def on_reshape(self, width, height):
        with self.program:
            sx = max((width / height) / self.aspect_ratio, 1.0)
            sy = max(self.aspect_ratio / (width / height), 1.0)
            trans_proj = mat4.orthographic((1.0 - sx) / 2, 1.0 - (1.0 - sx) / 2,
                                           1.0 - (1.0 - sy) / 2, (1.0 - sy) / 2)
            self.program.set_uniform_mat4('transProj', trans_proj)

        with self.seek_program:
            trans_proj = mat4.orthographic(0, 1, 0, height)
            self.seek_program.set_uniform_mat4('transProj', trans_proj)

    def save_annots(self):
        with self.clip_annot_path.open('w') as f:
            annots = {
                'session_id': self.session_id,
                'subject_id': self.subject_id,
                'clips': [
                    {'start_time': start_time, 'end_time': end_time}
                    for start_time, end_time in self.clips
                ]
            }
            json.dump(annots, f, indent=2)

    def add_clip(self, start_time, end_time):
        bisect.insort(self.clips, (round(start_time, 2), round(end_time, 2)))
        self.save_annots()

    def remove_clips(self, time):
        self.clips = [
            clip for clip in self.clips
            if not clip[0] <= time <= clip[1]
        ]
        self.save_annots()

    def on_close(self):
        # Pop the CUDA context created by PyCUDA.
        pycuda.autoinit.context.pop()

    def update(self, dt):
        if self.keyboard.was_pressed('['):
            self.annot_clip_start = self.cur_time
        if self.keyboard.was_pressed(']'):
            self.annot_clip_end = self.cur_time
            if self.annot_clip_start is not None and self.annot_clip_end is not None \
                    and self.annot_clip_end > self.annot_clip_start:
                self.add_clip(self.annot_clip_start, self.annot_clip_end)
                self.annot_clip_start = None
                self.annot_clip_end = None
        if self.keyboard.was_pressed(Key.DELETE):
            self.remove_clips(self.cur_time)
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
        if self.keyboard.was_pressed('\t'):
            self.cur_video_index = (self.cur_video_index + 1) % len(self.vls)
            self.next_time = -1
        if self.keyboard.was_pressed('1'):
            self.cur_video_index = 0
            self.next_time = -1
        if self.keyboard.was_pressed('2'):
            self.cur_video_index = 1
            self.next_time = -1
        if self.keyboard.was_pressed('3'):
            self.cur_video_index = 2
            self.next_time = -1
        if self.keyboard.was_pressed('.'):
            self.cur_time += 1 / self.vls[0].frame_rate
        if self.keyboard.was_pressed('>'):
            self.cur_time += 32 / self.vls[0].frame_rate
        if self.keyboard.was_pressed(','):
            self.cur_time -= 1 / self.vls[0].frame_rate
            self.next_time = -1
        if self.keyboard.was_pressed('<'):
            self.cur_time -= 32 / self.vls[0].frame_rate
            self.next_time = -1
        if not self.paused:
            self.cur_time += dt * self.playback_speed
        if self.playback_speed < 0:
            self.next_time = -1
        vl = self.vls[self.cur_video_index]
        duration = vl.duration - 0.1
        if self.keyboard.was_pressed(Key.HOME):
            self.cur_time = 0
            self.next_time = -1
        if self.keyboard.was_pressed(Key.END):
            self.cur_time = duration
            self.next_time = -1
        if self.keyboard.was_pressed(Key.PAGE_UP):
            for start_time, end_time in reversed(self.clips):
                if start_time < self.cur_time:
                    self.cur_time = start_time
                    self.next_time = -1
                    break
        if self.keyboard.was_pressed(Key.PAGE_DOWN):
            for start_time, end_time in self.clips:
                if start_time > self.cur_time:
                    self.cur_time = start_time
                    self.next_time = -1
                    break

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
            self.image = vl.read_frame()
            self.next_time = self.cur_time + 1 / vl.frame_rate
        else:
            while self.cur_time >= self.next_time:
                self.image = vl.read_frame()
                self.next_time += 1 / vl.frame_rate

        with self.seek_program:
            self.seek_program.set_uniform_float('progress', self.cur_time / duration)

    def render(self, dt):
        # Update the texture (using GPU operations).
        if self.image is not None:
            tensor = self.tex.tensor
            tensor[:, :, 3] = 255  # set alpha
            tensor[..., :3] = self.image.permute(1, 2, 0)[:tensor.shape[0], :tensor.shape[1]]
            in_clip = False
            clip_start_time = 0
            for start_clip, end_clip in self.clips:
                if start_clip <= self.cur_time <= end_clip:
                    in_clip = True
                    clip_start_time = start_clip
                    break
            if in_clip:
                rand = random.Random(int(clip_start_time * 1000))
                colour = (rand.randrange(256), rand.randrange(256), rand.randrange(256))
                for chan, value in enumerate(colour):
                    tensor[:4, :, chan].fill_(value)
                    tensor[:, :4, chan].fill_(value)
                    tensor[:, -4:, chan].fill_(value)
                    tensor[-4:, :, chan].fill_(value)
            self.tex.update()

        # Render the texture on a quad.
        with self.program, self.vao, self.tex.gl_texture:
            gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)

        with self.seek_program, self.vao_seek:
            gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)


if __name__=='__main__':
    torch.set_grad_enabled(False)
    VideoPlayer(parse_args()).run()
