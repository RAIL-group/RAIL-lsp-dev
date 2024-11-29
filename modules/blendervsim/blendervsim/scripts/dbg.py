from blendervsim import BlenderVSim
import numpy as np

import socket
import subprocess
import struct
import pickle

import matplotlib
matplotlib.use("TkAgg")  # Use TkAgg backend
import matplotlib.pyplot as plt

def main():
    # Start Blender as a subprocess
    with BlenderVSim(verbose=True) as blender:

        # Example messages to send
        messages_to_send = [
            {'command': 'render'},
            {'message': 'Hello from parent 2'},
            {'message': 'Hello from parent 2'},
            {'message': 'Hello from parent 2'},
            {'command': 'render'},
            # {'message': np.random.rand(2400, 2400, 3)},
            {'command': 'render'}
        ]
        for msg in messages_to_send:
            data = blender._send_receive_data(msg)
            print(msg, data)

    plt.imshow(data['rendered_image'])
    plt.show()

if __name__ == '__main__':
    main()
