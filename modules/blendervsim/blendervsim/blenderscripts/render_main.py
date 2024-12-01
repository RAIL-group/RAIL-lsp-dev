# Add to the path to import local packages
import bpy
import os
import sys

import bpy
import math
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from utils.manager import BlenderManager
from utils.build import add_map_data, _add_material_to_object, _apply_transforms

def set_top_down_orthographic_camera(camera, xbounds, ybounds):
    """
    Set the properties of a top-down orthographic camera in Blender to entirely observe
    a rectangular region defined by xbounds and ybounds.

    Parameters:
    camera (bpy.types.Object): The camera object.
    xbounds (tuple): A tuple of (xmin, xmax) defining the x boundaries of the region.
    ybounds (tuple): A tuple of (ymin, ymax) defining the y boundaries of the region.
    """
    if not camera or camera.type != 'CAMERA':
        raise ValueError("The provided object is not a camera.")

    xmin, xmax = xbounds
    ymin, ymax = ybounds

    # Calculate the center of the rectangular region
    center_x = (xmin + xmax) / 2.0
    center_y = (ymin + ymax) / 2.0

    # Calculate the orthographic scale
    width = xmax - xmin
    height = ymax - ymin
    ortho_scale = max(width, height) / 2.0

    # Set the camera properties
    camera.location = (center_x, center_y, 10.0)  # Top-down view, Z > 0
    camera.rotation_euler = (0.0, 0.0, 0.0)  # Ensure it's looking down
    camera.data.type = 'ORTHO'  # Set the camera to orthographic
    camera.data.ortho_scale = max(width, height)

    import bpy


def ensure_bounds_in_frame(camera, xbounds, ybounds, z=0.0):
    """
    [WIP: this function is still in development]

    Adjust the camera's settings to ensure the full bounds are in the frame,
    accounting for camera rotation.

    Parameters:
    camera (bpy.types.Object): The camera object.
    xbounds (tuple): (xmin, xmax) bounds of the region.
    ybounds (tuple): (ymin, ymax) bounds of the region.
    z (float): The Z-plane of the bounds (default is 0).
    """
    if not camera or camera.type != 'CAMERA':
        raise ValueError("The provided object is not a camera.")

    xmin, xmax = xbounds
    ymin, ymax = ybounds

    # Corners of the bounds
    corners = [
        (xmin, ymin, z),
        (xmin, ymax, z),
        (xmax, ymin, z),
        (xmax, ymax, z)
    ]

    # Get the camera world matrix
    cam_matrix_world = camera.matrix_world
    cam_matrix_world_inv = cam_matrix_world.inverted()

    # Transform corners into camera space
    import mathutils
    corners_cam_space = [cam_matrix_world_inv @ mathutils.Vector(corner) for corner in corners]

    # Get the extents of the corners in camera space
    x_extents = [c.x for c in corners_cam_space]
    y_extents = [c.y for c in corners_cam_space]

    # Adjust the camera's orthographic scale or FOV
    if camera.data.type == 'ORTHO':
        max_extent = max(max(x_extents) - min(x_extents), max(y_extents) - min(y_extents))
        camera.data.ortho_scale = max_extent
    elif camera.data.type == 'PERSP':
        max_x = max(abs(x) for x in x_extents)
        max_y = max(abs(y) for y in y_extents)
        distance = abs(min(c.z for c in corners_cam_space))  # Assuming bounds are in front of the camera
        camera.data.angle = 2 * np.arctan(max(max_x, max_y) / distance)



def create_rectangle_with_bounds(xbounds, ybounds, z, name, material_name):
    """
    Create a rectangle (plane) in Blender that matches the specified bounds,
    assigns a name to the object, and sets a material by name.

    Parameters:
    xbounds (tuple): A tuple of (xmin, xmax) defining the x boundaries of the rectangle.
    ybounds (tuple): A tuple of (ymin, ymax) defining the y boundaries of the rectangle.
    name (str): The name to assign to the rectangle object.
    material_name (str): The name of the material to assign to the rectangle.

    Returns:
    bpy.types.Object: The created plane object.
    """

    bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, 0))
    plane = bpy.context.active_object
    plane.scale.x = xbounds[1] - xbounds[0]
    plane.scale.y = ybounds[1] - ybounds[0]
    plane.location.x = sum(xbounds) / 2
    plane.location.y = sum(ybounds) / 2
    plane.location.z = z
    plane.name = name

    _add_material_to_object(plane, material_name)
    _apply_transforms(plane)

    return plane


class BlenderManagerOverhead(BlenderManager):
    def render_overhead(self,
                        map_data,
                        robot_poses=None,
                        observed_grid=None,
                        subgoal_data=None,
                        pixels_per_meter=None,
                        edge_buffer_meters=0.0,
                        render_settings=None):

        # Move the camera
        map_dim_x = map_data['resolution'] * map_data['semantic_grid'].shape[1]
        map_dim_y = map_data['resolution'] * map_data['semantic_grid'].shape[0]
        xbounds = [map_data['x_offset'] - edge_buffer_meters,
                   map_data['x_offset'] + edge_buffer_meters + map_dim_x]
        ybounds = [map_data['y_offset'] - edge_buffer_meters,
                   map_data['y_offset'] + edge_buffer_meters + map_dim_y]

        camera = bpy.data.objects.get('OrthoCamera')
        set_top_down_orthographic_camera(camera, xbounds, ybounds)

        if pixels_per_meter is not None:
            if render_settings is None: render_settings = dict()
            render_settings['resolution_x'] = round(pixels_per_meter * map_dim_x)
            render_settings['resolution_y'] = round(pixels_per_meter * map_dim_y)

        # Add the objects
        objects = add_map_data(map_data,
                               robot_poses,
                               observed_grid,
                               subgoal_data)
        objects.append(create_rectangle_with_bounds(xbounds, ybounds, -0.01, 'ground_tiled',
                                                    'ground_tiled'))

        # Render
        image = self.render_image(render_settings=render_settings)

        # Clean up: clear newly-added objects
        [bpy.data.objects.remove(obj, do_unlink=True) for obj in objects]
        [bpy.data.meshes.remove(mesh) for mesh in bpy.data.meshes if mesh.users == 0]
        [bpy.data.curves.remove(curve) for curve in bpy.data.curves if curve.users == 0]

        # Compute the image 'extent' from the camera parameters (more accurate)
        res_x = bpy.context.scene.render.resolution_x
        res_y = bpy.context.scene.render.resolution_y
        max_res = max(res_x, res_y)
        extent = [
            camera.location.x - camera.data.ortho_scale * res_x / max_res / 2,
            camera.location.x + camera.data.ortho_scale * res_x / max_res / 2,
            camera.location.y - camera.data.ortho_scale * res_y / max_res / 2,
            camera.location.y + camera.data.ortho_scale * res_y / max_res / 2,
        ]

        return image, {'extent': extent}

# # Example usage
# camera = bpy.data.objects.get("Camera")
# if camera:
#     xbounds = (-5, 5)
#     ybounds = (-3, 7)
#     ensure_bounds_in_frame(camera, xbounds, ybounds)



def main():
    with BlenderManagerOverhead() as manager:
        while manager.alive:
            manager.listen()


if __name__ == "__main__":
    main()
