import OpenGL.GL as gl
import numpy as np

from glupy.gl import VAO
from glupy.math import mat4
from posekit.gui.shaders import create_seekbar_shader


class OctagonalBone:
    def __init__(self, shader, start_pos, end_pos, colour=np.asarray([1.0, 1.0, 1.0, 1.0])):
        self.shader = shader
        self.colour = colour

        self.start_pos = start_pos
        self.end_pos = end_pos

        diff = self.end_pos - self.start_pos
        dist = np.linalg.norm(diff, 2)
        u = diff / (dist + 1e-6)

        axis = np.zeros(3)
        axis[np.argmin(np.abs(u))] = 1.0
        perp1 = np.cross(u, axis)
        perp2 = np.cross(u, perp1)

        vertex_data_fields = [('position', np.float32, 3)]

        a = 0.1 * diff + self.start_pos  # Mid-band location
        b = 0.1 * dist  # Thickness
        self.vertex_data = np.asarray([
            (tuple(self.start_pos),),
            (tuple(self.end_pos),),
            (tuple(a + b * perp1),),
            (tuple(a + b * perp2),),
            (tuple(a - b * perp1),),
            (tuple(a - b * perp2),),
        ], dtype=vertex_data_fields)

        self.index_data = np.asarray([
            3, 2, 0,
            1, 2, 3,
            4, 3, 0,
            1, 3, 4,
            5, 4, 0,
            1, 4, 5,
            2, 5, 0,
            1, 5, 2,
        ], dtype=np.uint32)

        self.vao = VAO()
        with self.vao:
            self.vbo = self.vao.create_vbo(shader, self.vertex_data)
            self.vbo.transfer_data_to_gpu(self.vertex_data)

            self.ebo = self.vao.create_ebo()
            self.ebo.transfer_data_to_gpu(self.index_data)

    def render(self, dt):
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        with self.shader, self.vao:
            self.shader.set_uniform_vec4('color', self.colour)
            gl.glDrawElements(gl.GL_TRIANGLES, len(self.index_data), gl.GL_UNSIGNED_INT, None)


class Ground:
    def __init__(self, shader, centre, normal, forwards, size=16000.0):
        self.shader = shader

        vertex_data_fields = [('position', np.float32, 3),
                              ('texcoord', np.float32, 2)]

        self.vertex_data = np.asarray([
            ((-1, 0, -1), (0, 0)),
            (( 1, 0, -1), (1, 0)),
            (( 1, 0,  1), (1, 1)),
            ((-1, 0,  1), (0, 1)),
        ], dtype=vertex_data_fields)

        # Calculate rotation matrix for the plane.
        up = forwards / np.linalg.norm(forwards, ord=2)
        c = normal / np.linalg.norm(normal, ord=2)
        a = np.cross(up, c)
        a = a / np.linalg.norm(a, ord=2)
        b = np.cross(c, a)
        rot = np.eye(4)
        rot[0, :3] = a
        rot[1, :3] = c
        rot[2, :3] = b

        model_matrix = mat4.identity()
        model_matrix = mat4.scale(size) @ model_matrix
        model_matrix = rot @ model_matrix
        model_matrix = mat4.translate(*centre) @ model_matrix
        with self.shader:
            self.shader.set_uniform_mat4('modelMatrix', model_matrix)

        self.index_data = np.asarray([
            0, 1, 2,
            0, 2, 3,
        ], dtype=np.uint32)

        self.vao = VAO()
        with self.vao:
            self.vbo = self.vao.create_vbo(shader, self.vertex_data)
            self.vbo.transfer_data_to_gpu(self.vertex_data)

            self.ebo = self.vao.create_ebo()
            self.ebo.transfer_data_to_gpu(self.index_data)

    def render(self, dt):
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        with self.shader, self.vao:
            gl.glDrawElements(gl.GL_TRIANGLES, len(self.index_data), gl.GL_UNSIGNED_INT, None)


class SeekBar:
    def __init__(self):
        self.shader = create_seekbar_shader()

        vertex_data = np.empty(4, [
            ('position', np.float32, 2),
            ('texcoord', np.float32, 2),
        ])

        bar_h = 10  # Seek bar height (in pixels)
        vertex_data['position'] = [(0, 0), (0, bar_h), (1, 0), (1, bar_h)]
        vertex_data['texcoord'] = [(0, 0), (0, 1), (1, 0), (1, 1)]

        self.vao = VAO()
        with self.vao:
            vbo = self.vao.create_vbo(self.shader, vertex_data)
            vbo.transfer_data_to_gpu(vertex_data)

    def set_progress(self, fraction):
        with self.shader:
            self.shader.set_uniform_float('progress', fraction)

    def on_reshape(self, width, height):
        with self.shader:
            trans_proj = mat4.orthographic(0, 1, 0, height)
            self.shader.set_uniform_mat4('transProj', trans_proj)

    def render(self, dt):
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        with self.shader, self.vao:
            gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)
