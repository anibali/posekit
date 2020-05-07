import OpenGL.GL as gl
import numpy as np

from glupy.gl import OpenGlApp, Key, ModifierKey, UniformBinding
from glupy.math import mat4
from posekit.gui.components import OctagonalBone
from posekit.gui.shaders import create_simple_shader

CAMERA_SPEED = 1.5
CAMERA_ZOOM_RANGE = [1000, 40000]


class PoseViewer(OpenGlApp):
    def __init__(self, joints_3d, skeleton):
        super().__init__('Pose viewer', 1600, 900, msaa=4)

        # Enable alpha blending.
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        # Enable back-face culling.
        gl.glEnable(gl.GL_CULL_FACE)
        gl.glCullFace(gl.GL_BACK)

        # Enable depth test.
        gl.glEnable(gl.GL_DEPTH_TEST)

        self.shader = create_simple_shader()

        self.trans_mats_binding = UniformBinding('transformMatrices', [
            ('viewMatrix', np.float32, (4, 4)),
            ('projMatrix', np.float32, (4, 4)),
        ])
        self.trans_mats_binding.bind_program(self.shader)
        self.trans_mats_ubo = self.trans_mats_binding.create_ubo(bind=True)
        self.trans_mats_ubo['viewMatrix'] = np.eye(4)
        self.trans_mats_ubo['projMatrix'] = np.eye(4)

        self.reset_free_camera()

        self.skeleton = skeleton
        self.bones = self.pose_to_bones(joints_3d)
        self.origin = joints_3d[skeleton.root_joint_id]

    def pose_to_bones(self, pose):
        pose = pose[..., :3]
        bones = []
        for j, p in enumerate(self.skeleton.joint_tree):
            if self.skeleton.joint_names[j].startswith('left'):
                colour = np.asarray([0.7, 0.7, 1.0, 1.0])
            elif self.skeleton.joint_names[j].startswith('right'):
                colour = np.asarray([1.0, 0.7, 0.7, 1.0])
            else:
                colour = np.asarray([1.0, 0.7, 1.0, 1.0])
            if j != p:
                bones.append(OctagonalBone(self.shader, pose[j], pose[p], colour))
        return bones

    def reset_free_camera(self):
        self.cam_azimuth = -np.pi / 2
        self.cam_elevation = np.pi / 2 + np.pi / 6
        self.cam_radius = 7000.0

    def use_free_camera(self):
        cam_x = self.cam_radius * np.sin(self.cam_elevation) * np.cos(self.cam_azimuth)
        cam_y = self.cam_radius * np.cos(self.cam_elevation)
        cam_z = self.cam_radius * np.sin(self.cam_elevation) * np.sin(self.cam_azimuth)
        trans_view = mat4.look_at(np.asarray([cam_x, cam_y, cam_z]) + self.origin, self.origin, np.asarray([0, 1, 0]))
        self.trans_mats_ubo['viewMatrix'] = trans_view
        trans_proj = mat4.perspective(np.pi / 3, self.window_width / self.window_height, 1.0, 1e6)
        self.trans_mats_ubo['projMatrix'] = trans_proj
        self.trans_mats_ubo.flush()

    def update(self, dt):
        camera_speed = CAMERA_SPEED
        # Use slower and more precise camera controls when the shift key is held down.
        if self.keyboard.has_modifier(ModifierKey.SHIFT):
            camera_speed /= 20

        if self.keyboard.is_down(Key.LEFT):
            self.cam_azimuth = np.fmod(self.cam_azimuth - camera_speed * dt, 2 * np.pi)
            self.snap_camera = -1
        if self.keyboard.is_down(Key.RIGHT):
            self.cam_azimuth = np.fmod(self.cam_azimuth + camera_speed * dt, 2 * np.pi)
            self.snap_camera = -1
        if self.keyboard.is_down(Key.DOWN):
            self.cam_elevation = np.fmax(self.cam_elevation - camera_speed * dt, 0.1)
            self.snap_camera = -1
        if self.keyboard.is_down(Key.UP):
            self.cam_elevation = np.fmin(self.cam_elevation + camera_speed * dt, np.pi - 0.1)
            self.snap_camera = -1
        if self.keyboard.is_down(Key.PAGE_UP):
            new_radius = (np.sqrt(self.cam_radius) - camera_speed * dt * 30) ** 2
            self.cam_radius = np.fmax(new_radius, CAMERA_ZOOM_RANGE[0])
            self.snap_camera = -1
        if self.keyboard.is_down(Key.PAGE_DOWN):
            new_radius = (np.sqrt(self.cam_radius) + camera_speed * dt * 30) ** 2
            self.cam_radius = np.fmin(new_radius, CAMERA_ZOOM_RANGE[1])
            self.snap_camera = -1
        if self.keyboard.is_down(Key.HOME):
            self.reset_free_camera()

    def render_bones(self, dt):
        for bone in self.bones:
            bone.render(dt)

    def render(self, dt):
        self.use_free_camera()
        self.render_bones(dt)
