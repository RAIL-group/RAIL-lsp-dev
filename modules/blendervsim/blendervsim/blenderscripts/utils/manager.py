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
        self.counter = 0

    def __enter__(self):
        self.sock.connect(("localhost", self.comm_port))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.sock.close()

    def listen(self):
        """Listen for data from parent and then execute command."""
        # Read message from parent
        input_data = receive_pickled_data(sys.stdin.buffer)
        self._sent_reply = False
        if input_data is None:
            return

        command = input_data.get('command')
        if command is None:
            input_data['status'] = 'Command not provided.'
            self._send(input_data)
        elif command == 'listen':
            raise ValueError('Command cannot be "listen".')
        elif command[0] == '_':
            raise ValueError('Command cannot begin with an underscore.')
        elif hasattr(self, command):
            # Call an arbitrary command
            method = getattr(self, command)
            if callable(method):
                return method(input_data)
            else:
                print(f"{command} is not callable.")
                self._send({'status': f'{command} is not callable'})
        else:
            print(f"{command} does not exist.")
            self._send({'status': 'No such command.'})

        # Some protection: a reply must be sent.
        if not self._sent_reply:
            raise RuntimeError("No reply sent from Blender. "
                               "All commands must have a reply.")

    def echo(self, data):
        print(f"Printing data: {data}")
        self._send({'status': 'done'})

    def _send(self, data):
        send_pickled_data(self.sock, data)
        self._sent_reply = True

    def close(self, _):
        self.alive = False
        self._send({'status', 'process closed'})

    def set_render_settings(self, data):
        # Set the render settings
        apply_render_settings(data)
        self.render_settings_set = True
        self._send({'status': 'set render settings'})

    def render_image(self, data):
        stime = time.time()
        if not self.render_settings_set or 'render_settings' in data.keys():
            apply_render_settings(data)
            self.render_settings_set = True

        # Render the image and load back in
        render_path = data.get('render_path', '/tmp/render_result.png')
        bpy.context.scene.render.filepath = render_path
        bpy.ops.render.render(write_still=True)
        image = np.asarray(Image.open(render_path))

        # Process the data (example: add a new key)
        out_data = {'status': 'rendered',
                    'rendered_image': image}

        # Send response back to parent
        self._send(out_data)
