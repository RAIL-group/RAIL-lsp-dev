import os
import subprocess
import pickle
import struct
import socket

BLENDER_EXE_PATH = '/blender/blender'
BLENDER_SCRIPT_PATH = '/modules/blendervsim/blenderscripts/render_gridmap_figure.py'


def _send_pickled_data(pipe, data):
    pickled_data = pickle.dumps(data)
    length = struct.pack('>I', len(pickled_data))
    pipe.write(length)
    pipe.write(pickled_data)
    pipe.flush()


def _receive_pickled_data_pipe(pipe):
    length_bytes = pipe.read(4)
    print("got it!", length_bytes)
    if not length_bytes:
        return None  # End of stream
    length = struct.unpack('>I', length_bytes)[0]
    print("got it!", length)
    pickled_data = pipe.read(length)
    print("DATA", pickled_data)
    return pickle.loads(pickled_data)


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


class BlenderVSim():
    def __init__(self, blender_exe_path=BLENDER_EXE_PATH,
                 blender_script_path=BLENDER_SCRIPT_PATH,
                 verbose=True):
        self.blender_exe_path = blender_exe_path
        self.blender_script_path = blender_script_path
        self.blender_comm_port = 9999
        self.verbose = verbose

        # Create a server socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def __enter__(self):
        # Start Blender as a subprocess
        self.server_socket.bind(("localhost", 9999))  # Bind to localhost and a port
        self.server_socket.listen(1)

        if self.verbose:
            stdout = subprocess.PIPE
        else:
            stdout = sys.stdout

        self.blender_process = subprocess.Popen(
            [self.blender_exe_path, '--background', '--python', self.blender_script_path],
            stdin=subprocess.PIPE,
            stdout=stdout,
            # stderr=subprocess.PIPE
        )

        self.conn, self.addr = self.server_socket.accept()
        return self

    def __exit__(self, type, value, traceback):
        # Close the subprocess
        self.blender_process.stdin.close()
        # self.blender_process.stdout.close()
        # self.blender_process.stderr.close()
        self.blender_process.wait()

    def send_receive_data(self, data):
        # Send message to Blender
        _send_pickled_data(self.blender_process.stdin, data)
        data = _receive_pickled_data(self.conn)
        return data

        # This doesn't always work...
        while True:
            # Read a line from stdout
            stdout_line = self.blender_process.stdout.readline()
            if stdout_line:
                print(f"[BLENDER OUT] {stdout_line.strip()}")

            # Read a line from stderr
            stderr_line = self.blender_process.stderr.readline()
            if stderr_line:
                print(f"[BLENDER ERR] {stderr_line.strip()}", file=sys.stderr)

            # Exit the loop when the process finishes and streams are empty
            if self.blender_process.poll() is not None and not stdout_line and not stderr_line:
                break
