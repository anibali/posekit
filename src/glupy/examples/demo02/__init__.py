import OpenGL.GL as gl
from importlib import resources
import numpy as np

import glupy.examples.demo02
from glupy import OpenGlApp, VAO, ShaderProgram
from glupy.math import mat4


class Demo02(OpenGlApp):
    def __init__(self):
        super().__init__('Demo #2', 1600, 900)

        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)

        vertex_code = resources.read_text(glupy.examples.demo02, 'demo02.vert')
        fragment_code = resources.read_text(glupy.examples.demo02, 'demo02.frag')
        self.program = ShaderProgram(vertex_code, fragment_code)

        vertex_data = np.empty(8, [
            ('position', np.float32, 3),
            ('color', np.float32, 4),
        ])
        index_data = np.empty(12 * 3, dtype=np.uint32)

        vertex_data['position'] = [
            (-1, -1, -1),
            (+1, -1, -1),
            (+1, +1, -1),
            (-1, +1, -1),
            (-1, -1, +1),
            (+1, -1, +1),
            (+1, +1, +1),
            (-1, +1, +1),
        ]
        vertex_data['color'] = [
            (0, 1, 0, 1),
            (1, 1, 0, 1),
            (1, 0, 0, 1),
            (0, 0, 1, 1),
            (1, 1, 1, 1),
            (1, 1, 1, 1),
            (1, 1, 1, 1),
            (1, 1, 1, 1),
        ]

        index_data[:] = [
            0, 1, 2,
            2, 3, 0,
            1, 5, 6,
            6, 2, 1,
            7, 6, 5,
            5, 4, 7,
            4, 0, 3,
            3, 7, 4,
            4, 5, 1,
            1, 0, 4,
            3, 2, 6,
            6, 7, 3
        ]

        self.vao = VAO()
        with self.vao:
            self.vbo = self.vao.create_vbo(self.program, vertex_data)
            self.vbo.transfer_data_to_gpu(vertex_data)

            self.ebo = self.vao.create_ebo()
            self.ebo.transfer_data_to_gpu(index_data)

        trans_model = mat4.scale(0.5) #@ mat4.rotate_axis_angle(0, 1/np.sqrt(2), 1/np.sqrt(2), np.pi / 4)
        trans_view = mat4.translate(2.0, 0, 5.0)
        # TODO: Adjust this when window is resized
        trans_proj = mat4.perspective(np.pi / 3, 16 / 9, 0.1, 100)

        with self.program:
            self.program.set_uniform_mat4('transModel', trans_model)
            self.program.set_uniform_mat4('transView', trans_view)
            self.program.set_uniform_mat4('transProj', trans_proj)

    def render(self, dt):
        with self.program, self.vao:
            gl.glDrawElements(gl.GL_TRIANGLES, 36, gl.GL_UNSIGNED_INT, None)


if __name__ == '__main__':
    Demo02().run()
