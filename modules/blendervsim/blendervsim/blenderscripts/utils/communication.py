import argparse
import pickle
import struct
import sys

# Important for ensuring parent receives all outputs/errors
sys.stderr.reconfigure(line_buffering=True)
sys.stdout.reconfigure(line_buffering=True)

def receive_pickled_data(pipe):
    length_bytes = pipe.read(4)
    if not length_bytes:
        return None  # End of stream
    length = struct.unpack('>I', length_bytes)[0]
    pickled_data = pipe.read(length)
    return pickle.loads(pickled_data)


def send_pickled_data(sock, data):
    pickled_data = pickle.dumps(data)
    length = struct.pack('>I', len(pickled_data))
    sock.sendall(length + pickled_data)


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
