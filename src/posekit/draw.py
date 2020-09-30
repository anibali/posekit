import PIL.Image
import PIL.ImageDraw
import numpy as np

from posekit.skeleton import Skeleton

_DEFAULT_COLOUR = (165, 165, 165)
_GROUP_COLOURS = {
    'left': (0, 0, 255),
    'right': (255, 0, 0),
    'centre': (255, 0, 255),
}


def _lighten_colour(colour):
    return tuple(map(lambda x: max(x, 165), colour))


def draw_pose_on_image_(
    joints_2d: np.ndarray,
    skeleton: Skeleton,
    image: PIL.Image.Image,
    point_radius: int = 2,
    visibilities=None,
):
    """Draw 2D pose on a PIL image.

    The pixel values of the image will be modified in-place.

    Args:
        joints_2d: The 2D joint locations (in pixels).
        skeleton: The skeleton description.
        image: The image to overlay the drawn pose on.
        point_radius: The radius of circles indicating joint locations. No circles will be drawn
            when ``point_radius=0``.
        visibilities: A mask describing which joints are visible.
    """
    joints_2d = np.asarray(joints_2d, dtype=np.int)
    draw = PIL.ImageDraw.Draw(image)
    # Determine colours.
    colours = [_GROUP_COLOURS.get(skeleton.get_joint_metadata(joint_id)['group'], _DEFAULT_COLOUR)
               for joint_id in range(skeleton.n_joints)]
    joint_colours = [*colours]
    limb_colours = [*colours]
    if visibilities is not None:
        for joint_id, parent_id in enumerate(skeleton.joint_tree):
            if not visibilities[joint_id]:
                joint_colours[joint_id] = _lighten_colour(colours[joint_id])
                limb_colours[joint_id] = _lighten_colour(colours[joint_id])
            if not visibilities[parent_id]:
                limb_colours[joint_id] = _lighten_colour(colours[joint_id])
    # Draw lines for limbs.
    for joint_id, (joint, colour) in enumerate(zip(joints_2d, limb_colours)):
        parent = joints_2d[skeleton.joint_tree[joint_id]]
        draw.line([*joint.tolist(), *parent.tolist()], fill=colour, width=1)
    # Draw points for joints.
    if point_radius > 0:
        for joint, colour in zip(joints_2d, joint_colours):
            draw.ellipse([*(joint - point_radius).tolist(), *(joint + point_radius).tolist()], fill=colour)
