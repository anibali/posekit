from importlib import resources

import posekit.gui.res
from glupy.gl import ShaderProgram


def create_simple_shader():
    return ShaderProgram(resources.read_text(posekit.gui.res, 'simple.vert'),
                         resources.read_text(posekit.gui.res, 'simple.frag'))


def create_checkerboard_shader():
    return ShaderProgram(resources.read_text(posekit.gui.res, 'checkerboard.vert'),
                         resources.read_text(posekit.gui.res, 'checkerboard.frag'))


def create_image_shader():
    return ShaderProgram(resources.read_text(posekit.gui.res, 'image.vert'),
                         resources.read_text(posekit.gui.res, 'image.frag'))


def create_seekbar_shader():
    return ShaderProgram(resources.read_text(posekit.gui.res, 'image.vert'),
                         resources.read_text(posekit.gui.res, 'seekbar.frag'))
