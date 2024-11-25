import subprocess
import pickle
import struct

BLENDER_EXE_PATH = '/blender/blender'
BLENDER_SCRIPT_PATH = '/modules/blendervsim/blenderscripts/render_gridmap_figure.py'


def _send_pickled_data(pipe, data):
    pickled_data = pickle.dumps(data)
    length = struct.pack('>I', len(pickled_data))
    pipe.write(length)
    pipe.write(pickled_data)
    pipe.flush()


def _receive_pickled_data(pipe):
    length_bytes = pipe.read(4)
    if not length_bytes:
        return None  # End of stream
    length = struct.unpack('>I', length_bytes)[0]
    pickled_data = pipe.read(length)
    return pickle.loads(pickled_data)


class BlenderVSim():
    def __init__(self, blender_exe_path=BLENDER_EXE_PATH,
                 blender_script_path=BLENDER_SCRIPT_PATH):
        self.blender_exe_path = blender_exe_path
        self.blender_script_path = blender_script_path

    def __enter__(self):
        # Start Blender as a subprocess
        self.blender_process = subprocess.Popen(
            [self.blender_exe_path, '--background', '--python', self.blender_script_path],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        return self

    def __exit__(self, type, value, traceback):
        # Close the subprocess
        self.blender_process.stdin.close()
        self.blender_process.stdout.close()
        self.blender_process.stderr.close()
        self.blender_process.wait()

    def send_receive_data(self, data):
        # Send message to Blender
        _send_pickled_data(self.blender_process.stdin, data)
        # Wait for response
        return _receive_pickled_data(self.blender_process.stdout)
