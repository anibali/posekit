from importlib import resources

import numpy as np

import glupy.examples.demo01
from glupy.gl import OpenGlApp, VAO, ShaderProgram, EBO, VBO


class Demo1(OpenGlApp):
    def __init__(self):
        super().__init__('Demo #1', 1600, 900)

        vertex_code = resources.read_text(glupy.examples.demo01, 'demo01.vert')
        fragment_code = resources.read_text(glupy.examples.demo01, 'demo01.frag')
        self.program = ShaderProgram(vertex_code, fragment_code)

        # # Option 1: Interleaved memory layout.
        # layout = np.dtype(([
        #     ('position', (np.float32, 3)),
        #     ('color', (np.float32, 4)),
        # ], 4), align=True)

        # Option 2: "Struct of arrays" memory layout.
        layout = np.dtype(([
            ('position', (np.float32, 3), 4),
            ('color', (np.float32, 4), 4),
        ]), align=True)

        vertex_data = np.empty(layout.shape, layout.base)
        vertex_data['position'] = [[-1, +1, 0], [+1, +1, 0], [-1, -1, 0], [+1, -1, 0]]
        vertex_data['color'] = [[0, 1, 0, 1], [1, 1, 0, 1], [1, 0, 0, 1], [0, 0, 1, 1]]

        index_data = np.asarray([
            0, 1, 2,
            3, 1, 2,
        ], dtype=np.uint32)

        self.vao = VAO(VBO(vertex_data), EBO(index_data), connect_to=self.program)

    def render(self, dt):
        with self.program, self.vao:
            self.vao.draw_elements()


if __name__ == '__main__':
    Demo1().run()
