from importlib import resources

import OpenGL.GL as gl
import numpy as np
import torch

import glupy.examples.demo03
from glupy.gl import VAO, ShaderProgram, OpenGlApp, VBO
from glupy.gl.torch import MappedTexture


class Demo3(OpenGlApp):
    def __init__(self):
        w, h = 512, 512
        super().__init__('Demo #3', w, h)

        # Ensure PyTorch CUDA is initialised (it is important that this happens _after_ PyCUDA
        # initialises its context, which is currently done via autoinit).
        assert torch.cuda.is_available()
        torch.empty(1, device='cuda')

        # Create a texture buffer with OpenGL and CUDA views
        self.tex = MappedTexture(h, w)

        vertex_code = resources.read_text(glupy.examples.demo03, 'demo03.vert')
        fragment_code = resources.read_text(glupy.examples.demo03, 'demo03.frag')
        self.program = ShaderProgram(vertex_code, fragment_code)

        vertex_data = np.empty(4, [
            ('position', np.float32, 2),
            ('texcoord', np.float32, 2),
        ])

        vertex_data['position'] = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
        vertex_data['texcoord'] = [(0, 0), (0, 1), (1, 0), (1, 1)]

        self.vao = VAO(vbo=VBO(vertex_data), connect_to=self.program)

        self.theta = 0

    def clean_up(self):
        del self.tex

    def render(self, dt):
        # Calculate the phase of the pulsating effect.
        self.theta = np.fmod(self.theta + 2 * dt, 2 * np.pi)

        # Update the texture (using GPU operations).
        with self.tex.modify() as tensor:
            tensor[:, :, 3] = 255  # set alpha
            tensor[:, :, 2] = round(255 * 0.5 * (np.sin(self.theta) + 1))  # set blue
            tensor[:128, :, 1] = 255  # horizontal green/cyan block
            tensor[:, :256, 0] = 255  # vertical red/magenta block

        # Render the texture on a quad.
        with self.program, self.vao, self.tex.gl_texture:
            gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)


if __name__=='__main__':
    torch.set_grad_enabled(False)
    Demo3().run()
