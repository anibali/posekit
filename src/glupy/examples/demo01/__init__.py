from importlib import resources

import OpenGL.GL as gl
import numpy as np

import glupy.examples.demo01
from glupy.gl import OpenGlApp, VAO, ShaderProgram


class Demo1(OpenGlApp):
    def __init__(self):
        super().__init__('Demo #1', 1600, 900)

        vertex_code = resources.read_text(glupy.examples.demo01, 'demo01.vert')
        fragment_code = resources.read_text(glupy.examples.demo01, 'demo01.frag')
        self.program = ShaderProgram(vertex_code, fragment_code)

        vertex_data = np.empty(4, [
            ('position', np.float32, 3),
            ('color', np.float32, 4),
        ])

        vertex_data['position'] = [(-1, +1, 0), (+1, +1, 0), (-1, -1, 0), (+1, -1, 0)]
        vertex_data['color'] = [(0, 1, 0, 1), (1, 1, 0, 1), (1, 0, 0, 1), (0, 0, 1, 1)]

        index_data = np.empty(2 * 3, dtype=np.uint32)
        index_data[:] = [
            0, 1, 2,
            3, 1, 2,
        ]

        self.vao = VAO()
        with self.vao:
            self.vbo = self.vao.create_vbo(self.program, vertex_data)
            self.vbo.transfer_data_to_gpu(vertex_data)
            self.ebo = self.vao.create_ebo()
            self.ebo.transfer_data_to_gpu(index_data)

    def render(self, dt):
        with self.program, self.vao:
            gl.glDrawElements(gl.GL_TRIANGLES, 6, gl.GL_UNSIGNED_INT, None)


if __name__ == '__main__':
    Demo1().run()
