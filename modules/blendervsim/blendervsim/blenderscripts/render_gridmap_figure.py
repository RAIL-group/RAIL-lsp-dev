import sys
import pickle
import struct
import numpy

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

def main():
    while True:
        # Read message from parent
        input_data = receive_pickled_data(sys.stdin.buffer)
        if input_data is None:
            break  # End of input stream
        print('Received in Blender:', input_data, file=sys.stderr)

        # Process the data (example: add a new key)
        input_data['reply'] = 'Hello from Blender'

        # Send response back to parent
        send_pickled_data(sys.stdout.buffer, input_data)

if __name__ == '__main__':
    main()
