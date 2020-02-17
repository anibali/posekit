from importlib import resources

import OpenGL.GL as gl
import numpy as np
import pycuda.autoinit
import pycuda.driver
import torch
from pycuda.gl import graphics_map_flags, RegisteredImage

import glupy.examples.demo03
from glupy.gl import VAO, ShaderProgram, OpenGlApp, Texture2d


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

        self.vao = VAO()
        with self.vao:
            self.vbo = self.vao.create_vbo(self.program, vertex_data)
            self.vbo.transfer_data_to_gpu(vertex_data)

        self.theta = 0

    def render(self, dt):
        # Calculate the phase of the pulsating effect.
        self.theta = np.fmod(self.theta + 2 * dt, 2 * np.pi)

        # Update the texture (using GPU operations).
        tensor = self.tex.tensor
        tensor[:, :, 3] = 255  # set alpha
        tensor[:, :, 2] = round(255 * 0.5 * (np.sin(self.theta) + 1))  # set blue
        tensor[:128, :, 1] = 255  # horizontal green/cyan block
        tensor[:, :256, 0] = 255  # vertical red/magenta block
        self.tex.update()

        # Render the texture on a quad.
        with self.program, self.vao, self.tex.gl_texture:
            gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)


if __name__=='__main__':
    torch.set_grad_enabled(False)
    Demo3().run()
