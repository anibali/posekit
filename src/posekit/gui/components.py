import OpenGL.GL as gl
import numpy as np

from glupy.gl import VAO, VBO, EBO
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
        vertex_data = np.asarray([
            (tuple(self.start_pos),),
            (tuple(self.end_pos),),
            (tuple(a + b * perp1),),
            (tuple(a + b * perp2),),
            (tuple(a - b * perp1),),
            (tuple(a - b * perp2),),
        ], dtype=vertex_data_fields)

        index_data = np.asarray([
            3, 2, 0,
            1, 2, 3,
            4, 3, 0,
            1, 3, 4,
            5, 4, 0,
            1, 4, 5,
            2, 5, 0,
            1, 5, 2,
        ], dtype=np.uint32)

        self.vao = VAO(vbo=VBO(vertex_data), ebo=EBO(index_data), connect_to=self.shader)

    def render(self, dt):
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        with self.shader, self.vao:
            self.shader.set_uniform_vec4('color', self.colour)
            self.vao.draw_elements()


class Ground:
    def __init__(self, shader, centre, normal, forwards, size=16000.0):
        self.shader = shader

        vertex_data_fields = [('position', np.float32, 3),
                              ('texcoord', np.float32, 2)]

        vertex_data = np.asarray([
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

        index_data = np.asarray([
            0, 1, 2,
            0, 2, 3,
        ], dtype=np.uint32)

        self.vao = VAO(vbo=VBO(vertex_data), ebo=EBO(index_data), connect_to=self.shader)

    def render(self, dt):
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        with self.shader, self.vao:
            self.vao.draw_elements()


class SeekBar:
    def __init__(self):
        self.height_px = 10  # Seek bar height (in pixels)

        self.shader = create_seekbar_shader()

        vertex_data = np.empty(4, [
            ('position', np.float32, 2),
            ('texcoord', np.float32, 2),
        ])

        vertex_data['position'] = [(0, 0), (0, 1), (1, 0), (1, 1)]
        vertex_data['texcoord'] = [(0, 0), (0, 1), (1, 0), (1, 1)]

        with self.shader:
            self.shader.set_uniform_float('depth', 0.0)

        self.vao = VAO(vbo=VBO(vertex_data), connect_to=self.shader)

    def set_progress(self, fraction):
        with self.shader:
            self.shader.set_uniform_float('progress', fraction)

    def on_reshape(self, width, height):
        with self.shader:
            offset = -(height - self.height_px) / height
            scale = height / self.height_px
            trans_proj = mat4.orthographic(0, 1, scale * (1 + offset), scale * offset)
            self.shader.set_uniform_mat4('transProj', trans_proj)

    def render(self, dt):
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        with self.shader, self.vao:
            gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)
