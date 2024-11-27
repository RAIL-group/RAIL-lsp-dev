from blendervsim import BlenderVSim
import numpy as np


import socket
import subprocess
import struct
import pickle

def _receive_pickled_data(sock):
    # Read 4 bytes for the length
    length_bytes = sock.recv(4)
    if not length_bytes:
        return None  # End of stream or closed socket

    length = struct.unpack('>I', length_bytes)[0]

    # Read the pickled data
    pickled_data = b''
    while len(pickled_data) < length:
        chunk = sock.recv(length - len(pickled_data))
        if not chunk:
            raise ConnectionError("Socket closed unexpectedly")
        pickled_data += chunk

    return pickle.loads(pickled_data)


# Create a server socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(("localhost", 9999))  # Bind to localhost and a port
server_socket.listen(1)

# Start the subprocess (Blender or another script)
BLENDER_EXE_PATH = '/blender/blender'
BLENDER_SCRIPT_PATH = '/modules/blendervsim/blenderscripts/render_gridmap_figure.py'
proc = subprocess.Popen([BLENDER_EXE_PATH, "--background", "--python", BLENDER_SCRIPT_PATH])

# Accept a connection from the subprocess
conn, addr = server_socket.accept()
print(f"Connection accepted from {addr}")

try:
    while True:
        data = _receive_pickled_data(conn)
        if data is None:
            break
        print(f"Received data: {data}")
finally:
    conn.close()
    server_socket.close()


def main():
    # Start Blender as a subprocess
    with BlenderVSim() as blender:
        import pickle
        print(pickle.dumps(np.zeros(10)))

        # Example messages to send
        messages_to_send = [
            {'message': 'Hello from parent 2'},
            {'message': np.random.rand(2400, 2400, 3)},
            {'message': 'Hello from parent 3'}
        ]
        for msg in messages_to_send:
            print(blender.send_receive_data(msg))


# if __name__ == '__main__':
#     main()
