import bpy
import time
import numpy as np
from PIL import Image
import socket
import sys

from .communication import send_pickled_data, receive_pickled_data, get_args
from .settings import apply_render_settings
from .build import add_map_data


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
        command = command_dict.get('command')
        args = command_dict.get('args', [])
        kwargs = command_dict.get('kwargs', {})
        if hasattr(self, command):
            method = getattr(self, command)
            if callable(method):
                self._send({'output': method(*args, **kwargs)})
                return

        raise AttributeError(f"{self} has no callable attribute '{command}'")

    def echo(self, message):
        print(f"Printing data: {message}")

    def set_render_settings(self, render_settings):
        apply_render_settings(render_settings)
        self.render_settings_set = True

    def render_image(self, render_path='/tmp/render_result.png', render_settings=None):
        if not self.render_settings_set or render_settings is not None:
            apply_render_settings(render_settings)
            self.render_settings_set = True

        # Render the image and load back in
        bpy.context.scene.render.filepath = render_path
        bpy.ops.render.render(write_still=True)
        return np.asarray(Image.open(render_path))
