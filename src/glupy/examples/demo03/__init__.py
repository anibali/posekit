#https://gist.github.com/victor-shepardson/5b3d3087dc2b4817b9bffdb8e87a57c4
from collections import namedtuple
from importlib import resources

import OpenGL.GL as gl
import numpy as np
import pycuda.autoinit
import pycuda.driver
import torch
from pycuda.gl import graphics_map_flags

import glupy.examples.demo03
from glupy import VAO, ShaderProgram, OpenGlApp


def copy_tensor_to_texture(tensor: torch.Tensor, cuda_buffer):
    # tensor shape: H x W x chans
    mapping = cuda_buffer.map()
    memcpy = pycuda.driver.Memcpy2D()
    memcpy.set_src_device(tensor.data_ptr())
    memcpy.set_dst_array(mapping.array(0, 0))
    nbytes = tensor.numel() * tensor.element_size()
    memcpy.height = tensor.shape[0]
    memcpy.width_in_bytes = memcpy.src_pitch = memcpy.dst_pitch = nbytes // memcpy.height
    memcpy(aligned=False)
    torch.cuda.synchronize(tensor.device)
    mapping.unmap()


def create_shared_texture(w, h, c=4,
        map_flags=graphics_map_flags.WRITE_DISCARD,  # Write-only from CUDA side
    ):
    """Create and return a Texture2D with OpenGL and CUDA views."""

    tex_target = gl.GL_TEXTURE_2D
    tex_handle = gl.glGenTextures(1)  # TODO: pair with glDeleteTextures
    gl.glBindTexture(tex_target, tex_handle)
    gl.glTexParameterf(tex_target, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
    gl.glTexParameterf(tex_target, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    gl.glTexParameterf(tex_target, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
    gl.glTexParameterf(tex_target, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
    gl.glTexParameterf(tex_target, gl.GL_TEXTURE_WRAP_R, gl.GL_CLAMP_TO_EDGE)
    gl.glTexImage2D(tex_target, 0, gl.GL_RGBA, w, h, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)
    gl.glBindTexture(tex_target, 0)
    tex = namedtuple('Texture2D', ['handle', 'target', 'nbytes', 'shape'])(tex_handle, tex_target, h * w * c, (h, w, c))

    cuda_buffer = pycuda.gl.RegisteredImage(int(tex.handle), tex.target, map_flags)
    return tex, cuda_buffer


class Demo3(OpenGlApp):
    def __init__(self):
        w, h = 512, 512
        super().__init__('Demo #3', w, h)

        # Ensure PyTorch CUDA is initialised.
        assert torch.cuda.is_available()
        torch.empty(1, device='cuda')

        # create a buffer with pycuda and gloo views
        self.tex, self.cuda_buffer = create_shared_texture(w, h, 4)

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

        self.x = 0

    def on_close(self):
        pycuda.autoinit.context.pop()

    def render(self, dt):
        self.x = np.fmod(self.x + 2 * dt, 2 * np.pi)

        # RGBA tensor with channels in texture order
        tensor = torch.zeros(self.tex.shape, dtype=torch.uint8, device='cuda')
        # tensor[:, :, 3] = 255  # set alpha
        tensor[:, :, 2] = round(255 * 0.5 * (np.sin(self.x) + 1))  # set blue
        tensor[:128, :, 1] = 255  # horizontal green/cyan block
        tensor[:, :256, 0] = 255  # vertical red/magenta block

        # Copy from PyTorch tensor into OpenGL texture.
        assert self.tex.nbytes == tensor.numel() * tensor.element_size()
        copy_tensor_to_texture(tensor, self.cuda_buffer)

        gl.glBindTexture(self.tex.target, int(self.tex.handle))
        with self.program, self.vao:
            gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)
        gl.glBindTexture(self.tex.target, 0)


if __name__=='__main__':
    torch.set_grad_enabled(False)
    Demo3().run()
