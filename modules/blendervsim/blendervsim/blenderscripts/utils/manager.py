import bpy
import time
import numpy as np
from PIL import Image
import socket
import sys

from .communication import send_pickled_data, receive_pickled_data, get_args
from .settings import apply_render_settings
from .build import (add_map_data, _add_material_to_object, _apply_transforms,
                    set_top_down_orthographic_camera, create_rectangle_with_bounds)


DEFAULT_RENDER_SETTINGS = {
    "engine": "CYCLES",
    "device": "CPU",
    "resolution_x": 256,
    "resolution_y": 256,
    "samples": 1,
    "use_adaptive_sampling": True,
    "use_denoising": False,
}


class BlenderManager(object):
    def __init__(self, comm_port=None):
        if comm_port is None:
            args = get_args()
            self.comm_port = args.comm_port
        else:
            self.comm_port = comm_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.alive = True
        self.render_settings_set = False

    def __enter__(self):
        self.sock.connect(("localhost", self.comm_port))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.sock.close()

    def _send(self, data):
        send_pickled_data(self.sock, data)

    def close(self):
        print("Blender shutting down!")
        self.alive = False

    def listen(self):
        """
        Listen for data from parent and then execute command.
        """

        # Read message from parent
        command_dict = receive_pickled_data(sys.stdin.buffer)
        if command_dict is None:
            return

        # Look for the function 'command' and then pass args/kwargs to it
        command = command_dict.get("command")
        args = command_dict.get("args", [])
        kwargs = command_dict.get("kwargs", {})
        if hasattr(self, command):
            method = getattr(self, command)
            if callable(method):
                self._send({"output": method(*args, **kwargs)})
                return

        raise AttributeError(f"{self} has no callable attribute '{command}'")

    def echo(self, message):
        print(f"Printing data: {message}")

    def set_render_settings(self, render_settings):
        apply_render_settings(render_settings)
        self.render_settings_set = True

    def render_image(self, render_path="/tmp/render_result.png", render_settings=None):
        if not self.render_settings_set or render_settings is not None:
            apply_render_settings(render_settings)
            self.render_settings_set = True

        # Render the image and load back in
        bpy.context.scene.render.filepath = render_path
        bpy.ops.render.render(write_still=True)
        return np.asarray(Image.open(render_path))

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
