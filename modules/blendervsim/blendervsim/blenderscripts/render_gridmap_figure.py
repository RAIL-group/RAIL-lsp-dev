# Add to the path to import local packages
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import bpy
import numpy as np
from utils.communication import send_pickled_data, receive_pickled_data
from PIL import Image

import socket
import struct
import pickle
import sys
import argparse

def get_args():
    # Find where '--' occurs in sys.argv
    try:
        idx = sys.argv.index("--")
        # Only take arguments after '--'
        args = sys.argv[idx + 1:]
    except ValueError:
        # If '--' is not found, no arguments are passed to the script
        args = []

    # Set up argparse
    parser = argparse.ArgumentParser(description="Blender Visual Simulator.")
    parser.add_argument("--comm-port", type=int, help="Port over which Blender sends data.")
    parsed_args = parser.parse_args(args)
    return parsed_args



def apply_render_settings(settings):
    """Apply render settings from a dictionary."""
    scene = bpy.context.scene
    render = scene.render  # Render settings
    render.image_settings.file_format = 'PNG'

    # Set general render settings
    for key, value in settings.items():
        if hasattr(render, key):
            setattr(render, key, value)
        elif render.engine == "CYCLES" and hasattr(bpy.context.scene.cycles, key):
            setattr(bpy.context.scene.cycles, key, value)
        else:
            print(f"Warning: Unknown setting {key}")

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

    def __enter__(self):
        self.sock.connect(("localhost", self.comm_port))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.sock.close()

    def listen(self):
        """Listen for data from parent and then execute command."""
        # Read message from parent
        input_data = receive_pickled_data(sys.stdin.buffer)
        if input_data is None:
            return

        command = input_data.get('command')
        if command is None:
            self._send({'status': 'command not provided'})
        elif command == 'listen':
            raise ValueError('Command cannot be "listen".')
        elif command[0] == '_':
            raise ValueError('Command cannot begin with an underscore.')
        elif hasattr(obj, command):
            # Call an arbitrary command
            method = getattr(obj, command)
            if callable(method):
                return method(*args, **kwargs)
            else:
                print(f"{command} is not callable.")
                self._send({'status': f'{command} is not callable'})
        else:
            print(f"{command} does not exist.")
            self._send({'status': 'No such command.'})
        return None

    def _send(self, data):
        send_pickled_data(self.sock, data)

    def close(self, _):
        self.alive = False
        self._send({'status', 'done'})

def main():

    print("IN")
    with BlenderManager() as manager:
        print("loop")
        while manager.alive:
            manager.listen()

    return

    # Get arguments (including comm-port)
    args = get_args()

    # Create a client socket and connect to the parent process
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(("localhost", args.comm_port))

    while True:
        # Read message from parent
        input_data = receive_pickled_data(sys.stdin.buffer)
        if input_data is None:
            continue

        if input_data.get('command') == 'close':
            send_pickled_data(sock, {'status', 'done'})
            break

        # Set the render settings
        render_settings = DEFAULT_RENDER_SETTINGS.copy()
        if 'render_settings' in input_data.keys():
            for k, v in input_data['render_settings']:
                render_settings[k] = v
        apply_render_settings(render_settings)

        # # Set the render engine and output settings
        # bpy.context.scene.render.image_settings.file_format = 'PNG'
        # bpy.context.scene.render.engine = 'CYCLES'  # Or 'BLENDER_EEVEE'
        # bpy.context.scene.cycles.device = 'CPU'
        # bpy.context.scene.cycles.samples = 1  # Adjust sample count for quality
        # bpy.context.scene.cycles.use_adaptive_sampling = True
        # bpy.context.scene.cycles.use_denoising = False

        # Render the image and load back in
        output_path = "/tmp/render_result.png"  # Adjust to your desired location
        bpy.context.scene.render.filepath = output_path
        bpy.ops.render.render(write_still=True)
        image = np.asarray(Image.open(output_path))

        # Process the data (example: add a new key)
        out_data = {'status': 'rendered',
                    'rendered_image': image}

        # Send response back to parent
        send_pickled_data(sock, out_data)

if __name__ == "__main__":
    main()
