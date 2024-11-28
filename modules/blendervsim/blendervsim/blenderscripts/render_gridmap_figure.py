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

def _send_pickled_data(sock, data):
    pickled_data = pickle.dumps(data)
    length = struct.pack('>I', len(pickled_data))
    sock.sendall(length + pickled_data)

# Create a client socket and connect to the parent process
# sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# sock.connect(("localhost", 9999))

# try:
#     # Example: Send data
#     send_pickled_data(sock, {"message": "Hello from Blender"})
# finally:
#     sock.close()


# # Create a client socket and connect to the parent process
# sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# sock.connect(("localhost", 9999))


def main():
    # Create a client socket and connect to the parent process
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(("localhost", 9999))
    import sys

    while True:
        # Read message from parent
        input_data = receive_pickled_data(sys.stdin.buffer)
        if input_data is None:
            continue

        # Set the render engine and output settings
        bpy.context.scene.render.engine = 'CYCLES'  # Or 'BLENDER_EEVEE'
        bpy.context.scene.cycles.device = 'CPU'
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.cycles.samples = 1  # Adjust sample count for quality
        bpy.context.scene.cycles.use_adaptive_sampling = True
        bpy.context.scene.cycles.use_denoising = False

        # Render the scene
        output_path = "/tmp/render_result.png"  # Adjust to your desired location
        bpy.context.scene.render.filepath = output_path

        import sys
        import os
        from contextlib import redirect_stdout

        # with open(os.devnull, 'w') as fnull:
        #     with redirect_stdout(fnull):
        #         # Perform render operations here
        #         bpy.ops.render.render(write_still=True)

        bpy.ops.render.render(write_still=True)
        # send_pickled_data(sys.stdout.buffer, input_data)

        # Load the rendered image
        image = np.asarray(Image.open(output_path))

        # Process the data (example: add a new key)
        input_data['reply'] = 'Hello from Blender'
        input_data['image_type'] = type(image)
        input_data['rendered_image'] = image

        # Send response back to parent
        _send_pickled_data(sock, input_data)

main()
