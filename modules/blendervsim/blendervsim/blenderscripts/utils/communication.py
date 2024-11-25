import pickle
import struct
import numpy as np

def send_pickled_data(pipe, data):
    pickled_data = pickle.dumps(data)
    length = struct.pack('>I', len(pickled_data))
    pipe.write(length)
    pipe.write(pickled_data)
    pipe.flush()

def receive_pickled_data(pipe):
    length_bytes = pipe.read(4)
    if not length_bytes:
        return None  # End of stream
    length = struct.unpack('>I', length_bytes)[0]
    pickled_data = pipe.read(length)
    return pickle.loads(pickled_data)
