from importlib import resources

import OpenGL.GL as gl
import numpy as np

import glupy.examples.demo04
from glupy import OpenGlApp, VAO, ShaderProgram, Key
from glupy.math import mat4
from posekit.skeleton import skeleton_registry

POSE = np.asarray(
    [[-3596.77759359,  1440.20459288, 13889.1501241 ],
     [-3674.02700896,   957.76349163, 13897.64851594],
     [-4069.29373748,   638.94500969, 13950.89930343],
     [-4020.92375082,   666.8599503 , 14254.54295457],
     [-4164.62511548,  1136.5902153 , 14128.65783666],
     [-4371.23650458,  1511.68095123, 14157.22505715],
     [-4040.52738576,   649.39606676, 14113.11860944],
     [-3980.65464075,   190.08999595, 14098.63679296],
     [-3957.10272226,    96.67068839, 14094.20865689],
     [-3881.62183779,  -174.06185146, 14086.25464788],
     [-3741.67715483,   518.83745453, 13722.81112096],
     [-4011.8833725 ,   452.81546631, 13738.52577327],
     [-4062.57389269,   193.65792606, 13855.55163398],
     [-3899.21030149,   191.90752373, 14374.98988218],
     [-3826.67297069,   444.69780302, 14416.19540235],
     [-3630.39824821,   515.20018004, 14175.38611995],],
    dtype=np.float32
)


class OctagonalBone:
    def __init__(self, shader_program, start_pos, end_pos, colour=np.asarray([1.0, 1.0, 1.0, 1.0])):
        self.program = shader_program
        self.colour = colour

        self.start_pos = start_pos
        self.end_pos = end_pos

        diff = self.end_pos - self.start_pos
        dist = np.linalg.norm(diff, 2)
        u = diff / dist

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
            self.vbo = self.vao.create_vbo(shader_program, self.vertex_data)
            self.vbo.transfer_data_to_gpu(self.vertex_data)

            self.ebo = self.vao.create_ebo()
            self.ebo.transfer_data_to_gpu(self.index_data)

    def render(self, dt):
        with self.program, self.vao:
            self.program.set_uniform_vec4('color', self.colour)
            gl.glDrawElements(gl.GL_TRIANGLES, len(self.index_data), gl.GL_UNSIGNED_INT, None)


class Demo04(OpenGlApp):
    def __init__(self):
        super().__init__('Demo #4', 1600, 900)

        # Enable alpha blending.
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        # Enable back-face culling.
        gl.glEnable(gl.GL_CULL_FACE)
        gl.glCullFace(gl.GL_BACK)

        # Turn on anti-aliased wireframe mode.
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        gl.glEnable(gl.GL_LINE_SMOOTH)

        vertex_code = resources.read_text(glupy.examples.demo04, 'demo04.vert')
        fragment_code = resources.read_text(glupy.examples.demo04, 'demo04.frag')
        self.program = ShaderProgram(vertex_code, fragment_code)

        self.reset_camera()

        skeleton = skeleton_registry['mpii_16j']
        pose = (POSE - POSE[skeleton.root_joint_id])

        self.bones = []
        for j, p in enumerate(skeleton.joint_tree):
            if j != p:
                self.bones.append(OctagonalBone(self.program, pose[j], pose[p]))

        self.bones.append(OctagonalBone(self.program, np.zeros(3), np.asarray([200.0, 0, 0]), np.asarray([1.0, 0.0, 0.0, 1.0])))
        self.bones.append(OctagonalBone(self.program, np.zeros(3), np.asarray([0, 200.0, 0]), np.asarray([0.0, 1.0, 0.0, 1.0])))
        self.bones.append(OctagonalBone(self.program, np.zeros(3), np.asarray([0, 0, 200.0]), np.asarray([0.0, 0.0, 1.0, 1.0])))

        self.on_reshape(self.window_width, self.window_height)

    def reset_camera(self):
        self.cam_azimuth = -np.pi / 2
        self.cam_elevation = np.pi / 2
        self.cam_radius = 2000.0

    def on_reshape(self, width, height):
        trans_proj = mat4.perspective(np.pi / 3, width / height, 1.0, 1e6)
        with self.program:
            self.program.set_uniform_mat4('transProj', trans_proj)

    def update(self, dt):
        camera_speed = 1.5
        if self.keyboard.is_down(Key.LEFT):
            self.cam_azimuth = np.fmod(self.cam_azimuth - camera_speed * dt, 2 * np.pi)
        if self.keyboard.is_down(Key.RIGHT):
            self.cam_azimuth = np.fmod(self.cam_azimuth + camera_speed * dt, 2 * np.pi)
        if self.keyboard.is_down(Key.DOWN):
            self.cam_elevation = np.fmax(self.cam_elevation - camera_speed * dt, 0.1)
        if self.keyboard.is_down(Key.UP):
            self.cam_elevation = np.fmin(self.cam_elevation + camera_speed * dt, np.pi - 0.1)
        if self.keyboard.is_down(Key.PAGE_UP):
            self.cam_radius = np.fmax((np.sqrt(self.cam_radius) - camera_speed * dt * 30) ** 2, 1000)
        if self.keyboard.is_down(Key.PAGE_DOWN):
            self.cam_radius = np.fmin((np.sqrt(self.cam_radius) + camera_speed * dt * 30) ** 2, 20000)
        if self.keyboard.is_down(Key.HOME):
            self.reset_camera()

    def render(self, dt):
        cam_x = self.cam_radius * np.sin(self.cam_elevation) * np.cos(self.cam_azimuth)
        cam_y = self.cam_radius * np.cos(self.cam_elevation)
        cam_z = self.cam_radius * np.sin(self.cam_elevation) * np.sin(self.cam_azimuth)
        trans_view = mat4.look_at(np.asarray([cam_x, cam_y, cam_z]), np.asarray([0, 0, 0]), np.asarray([0, 1, 0]))

        with self.program:
            self.program.set_uniform_mat4('transView', trans_view)

        for bone in self.bones:
            bone.render(dt)


if __name__ == '__main__':
    Demo04().run()
