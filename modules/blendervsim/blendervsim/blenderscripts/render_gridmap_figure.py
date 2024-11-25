# Add to the path to import local packages
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import bpy
import numpy as np
from utils.communication import send_pickled_data, receive_pickled_data

def main():
    while True:
        # Read message from parent
        input_data = receive_pickled_data(sys.stdin.buffer)
        if input_data is None:
            break  # End of input stream

        # Process the data (example: add a new key)
        input_data['reply'] = 'Hello from Blender'

        # Send response back to parent
        send_pickled_data(sys.stdout.buffer, input_data)

if __name__ == '__main__':
    main()
