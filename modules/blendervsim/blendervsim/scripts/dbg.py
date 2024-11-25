import subprocess
import pickle
import struct
from blendervsim import BlenderVSim

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
    # Start Blender as a subprocess
    process = subprocess.Popen(
        ['/blender/blender', '--background', '--python', '/modules/blendervsim/blenderscripts/render_gridmap_figure.py'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    blender_constructor = BlenderVSim()
    with BlenderVSim() as blender:

        # Example messages to send
        messages_to_send = [
            {'message': 'Hello from parent 1'},
            {'message': 'Hello from parent 2'},
            {'message': 'Hello from parent 3'}
        ]
        for msg in messages_to_send:
            print(blender.send_receive_data(msg))


    return
    for msg in messages_to_send:
        # Send message to Blender
        send_pickled_data(process.stdin, msg)

        # Wait for response
        response = receive_pickled_data(process.stdout)
        if response is None:
            print("No response received. Exiting.")
            break
        print('Received from Blender (dbg):', response)

    # Close the subprocess
    process.stdin.close()
    process.stdout.close()
    process.stderr.close()
    process.wait()

    blender = B

if __name__ == '__main__':
    main()
