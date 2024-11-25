from blendervsim import BlenderVSim
import numpy as np


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


if __name__ == '__main__':
    main()
