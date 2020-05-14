import OpenGL.GL as gl
import numpy as np

from glupy.gl import VAO, VBO
from glupy.gl.torch import MappedTexture
from glupy.math import mat4
from posekit.gui.shaders import create_image_shader


class OrthImage:
    def __init__(self, width, height):
        self.shader = create_image_shader()

        vertex_data = np.empty(4, [
            ('position', np.float32, 2),
            ('texcoord', np.float32, 2),
        ])

        vertex_data['position'] = [(0, 0), (0, 1), (1, 0), (1, 1)]
        vertex_data['texcoord'] = [(0, 0), (0, 1), (1, 0), (1, 1)]

        self.vao = VAO(vbo=VBO(self.shader, vertex_data.dtype))
        with self.vao:
            self.vao.vbo.connect_vertex_attributes()
            self.vao.vbo.transfer_data_to_gpu(vertex_data)

        self.image = None
        self.tex = MappedTexture(height, width)
        self.aspect_ratio = width / height

    def set_image(self, image):
        self.image = image
        with self.tex.modify() as tensor:
            tensor[:, :, 3] = 255  # set alpha
            tensor[..., :3] = self.image.permute(1, 2, 0)[:tensor.shape[0], :tensor.shape[1]]

    def on_reshape(self, width, height):
        with self.shader:
            sx = max((width / height) / self.aspect_ratio, 1.0)
            sy = max(self.aspect_ratio / (width / height), 1.0)
            trans_proj = mat4.orthographic((1.0 - sx) / 2, 1.0 - (1.0 - sx) / 2,
                                           1.0 - (1.0 - sy) / 2, (1.0 - sy) / 2)
            self.shader.set_uniform_mat4('transProj', trans_proj)

    def render(self, dt):
        if self.image is None:
            return
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        with self.shader, self.vao, self.tex.gl_texture:
            gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)
