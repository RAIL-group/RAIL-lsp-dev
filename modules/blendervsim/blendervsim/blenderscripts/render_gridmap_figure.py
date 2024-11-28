# Add to the path to import local packages
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import bpy
import numpy as np
from utils.communication import receive_pickled_data
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


def _send_pickled_data(sock, data):
    pickled_data = pickle.dumps(data)
    length = struct.pack('>I', len(pickled_data))
    sock.sendall(length + pickled_data)


def main():
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
            _send_pickled_data(sock, {'status', 'done'})
            break

        # Set the render engine and output settings
        bpy.context.scene.render.engine = 'CYCLES'  # Or 'BLENDER_EEVEE'
        bpy.context.scene.cycles.device = 'CPU'
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.cycles.samples = 1  # Adjust sample count for quality
        bpy.context.scene.cycles.use_adaptive_sampling = True
        bpy.context.scene.cycles.use_denoising = False

        # Render the image and load back in
        output_path = "/tmp/render_result.png"  # Adjust to your desired location
        bpy.context.scene.render.filepath = output_path
        bpy.ops.render.render(write_still=True)
        image = np.asarray(Image.open(output_path))

        # Process the data (example: add a new key)
        out_data = {'status': 'rendered',
                    'rendered_image': image}

        # Send response back to parent
        _send_pickled_data(sock, out_data)

if __name__ == "__main__":
    main()
