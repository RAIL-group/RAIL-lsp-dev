import os
import subprocess
import pickle
import struct
import socket
import sys

BLENDER_EXE_PATH = '/blender/blender'
BLENDER_SCRIPT_PATH = '/modules/blendervsim/blenderscripts/render_gridmap_figure.py'


async def _send_pickled_data(pipe, data):
    print('data', data)
    pickled_data = pickle.dumps(data)
    length = struct.pack('>I', len(pickled_data))
    pipe.write(length)
    pipe.write(pickled_data)
    # pipe.flush()
    await pipe.drain()


def _receive_pickled_data_pipe(pipe):
    length_bytes = pipe.read(4)
    if not length_bytes:
        return None  # End of stream
    length = struct.unpack('>I', length_bytes)[0]
    pickled_data = pipe.read(length)
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
        self.blender_comm_port = None
        self.verbose = verbose

        # Create a server socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def __enter__(self):
        # Start Blender as a subprocess
        self.server_socket.bind(("localhost", 0))  # Bind to localhost and an available port
        self.blender_comm_port = self.server_socket.getsockname()[1]  # Get the assigned port number
        self.server_socket.listen(1)

        if self.verbose:
            stdout = sys.stdout
        else:
            stdout = subprocess.PIPE

        self.blender_process = subprocess.Popen(
            [self.blender_exe_path, '--background', '--python', self.blender_script_path,
             '--', '--comm-port', str(self.blender_comm_port)],
            stdin=subprocess.PIPE,
            stdout=stdout,
            # stderr=subprocess.PIPE
        )

        self.conn, self.addr = self.server_socket.accept()
        return self

    def __exit__(self, type, value, traceback):
        # Close the subprocess
        self.close_blender()
        self.blender_process.stdin.close()
        self.blender_process.wait()
        # self.blender_process.stdout.close()
        # self.blender_process.stderr.close()

    def _send_receive_data(self, data):
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

    def close_blender(self):
        try:
            _send_pickled_data(self.blender_process.stdin, {'command': 'close'})
        except e:
            pass

    async def _read_stream(self, stream, callback):
        """Reads lines from a stream and sends them to the callback."""
        while True:
            line = await stream.readline()
            if not line:  # End of stream
                break
            await callback(line.decode().strip())

    async def _read_stdout(self):
        """Process data from stdout."""
        async def handle_data(data):
            print(f"[Blender Out] {data}")  # Handle data here
            # Process the received data (e.g., append to a list or parse)

        await self._read_stream(self.blender_process.stdout, handle_data)

    async def _read_stderr(self):
        """Process data from stderr."""
        async def handle_error(data):
            print(f"[Blender Error] {data}")  # Handle error here
            raise RuntimeError(f"Subprocess error: {data}")

        await self._read_stream(self.blender_process.stderr, handle_error)

# ################### Dragons

import asyncio
import socket
import subprocess
import sys
import struct
import pickle

class BlenderVSim:
    def __init__(self, blender_exe_path=BLENDER_EXE_PATH,
                 blender_script_path=BLENDER_SCRIPT_PATH,
                 verbose=True):
        self.blender_exe_path = blender_exe_path
        self.blender_script_path = blender_script_path
        self.blender_comm_port = None
        self.verbose = verbose
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.loop = asyncio.new_event_loop()
        self._stdout_task = None
        self._stderr_task = None
        self._stderr_logs = ''

    def __enter__(self):
        asyncio.set_event_loop(self.loop)
        return self.loop.run_until_complete(self._async_enter())

    def __exit__(self, exc_type, exc_value, traceback):
        self.loop.run_until_complete(self._async_exit())
        self.loop.close()

    async def _async_enter(self):
        self.server_socket.bind(("localhost", 0))
        self.blender_comm_port = self.server_socket.getsockname()[1]
        self.server_socket.listen(1)

        self.blender_process = await asyncio.create_subprocess_exec(
            self.blender_exe_path, '--background', '--python', self.blender_script_path,
            '--', '--comm-port', str(self.blender_comm_port),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        loop = asyncio.get_event_loop()
        self.conn, self.addr = await loop.run_in_executor(None, self.server_socket.accept)

        # Start listening to stdout and stderr asynchronously
        self._stdout_task = loop.create_task(self._read_stdout())
        self._stderr_task = loop.create_task(self._read_stderr())

        # try:
        #     await asyncio.gather(self._stdout_task, self._stderr_task)
        # except Exception as e:
        #     raise e

        return self

    async def _async_exit(self):
        # Cancel stdout and stderr listeners
        if self._stdout_task:
            self._stdout_task.cancel()
        if self._stderr_task:
            self._stderr_task.cancel()

        # Send a message to blender to close
        await _send_pickled_data(self.blender_process.stdin, {'command': 'close'})
        self.blender_process.stdin.close()
        await self.blender_process.wait()

    async def _read_stdout(self):
        """Asynchronously read stdout."""
        while True:
            line = await self.blender_process.stdout.readline()
            if not line:
                break
            if self.verbose:
                print(f"[Blender stdout] {line.decode().strip()}")  # Replace with custom processing logic

    async def _read_stderr(self):
        """Asynchronously read stderr."""
        while True:
            line = await self.blender_process.stderr.readline()
            if not line:
                break
            # print(f"STDERR: {line.decode().strip()}")  # Replace with custom error handling
            self._stderr_logs += f"[Blender Error] {line.decode().strip()}\n"

    def _send_receive_data(self, data):
        """Synchronous interface for sending and receiving data."""
        try:
            return self.loop.run_until_complete(self._async_send_receive_data(data))
        except (ConnectionResetError, BrokenPipeError) as e:
            error_msg = self._stderr_logs
            self._stderr_logs = ''
            raise RuntimeError(f"Blender has died with the following error: {e}\n\n{error_msg}")

    async def _async_send_receive_data(self, data):
        """Asynchronous method for sending and receiving data."""
        # Send data to Blender
        await _send_pickled_data(self.blender_process.stdin, data)
        # Receive response
        loop = asyncio.get_event_loop()
        received_data = await loop.run_in_executor(None, _receive_pickled_data, self.conn)
        return received_data


# Supporting functions for pickle-based communication
def _send_pickled_data(stream, data):
    """Serialize the data with pickle and send it over the stream."""
    pickled_data = pickle.dumps(data)
    length = struct.pack('>I', len(pickled_data))
    stream.write(length + pickled_data)
    stream.flush()  # Ensure data is sent immediately

def _receive_pickled_data(conn):
    """Receive pickled data from a socket."""
    # Read the length of the incoming message
    length_bytes = conn.recv(4)
    if not length_bytes:
        return None  # End of stream
    length = struct.unpack('>I', length_bytes)[0]

    # Read the actual data
    data = b""
    while len(data) < length:
        packet = conn.recv(length - len(data))
        if not packet:
            raise ConnectionError("Connection closed before receiving all data")
        data += packet

    return pickle.loads(data)

async def _send_pickled_data(stream, data):
    """Serialize the data with pickle and send it over an asyncio stream."""
    # Serialize the data
    pickled_data = pickle.dumps(data)
    # Prepend the length of the serialized data (4 bytes, big-endian)
    length = struct.pack('>I', len(pickled_data))
    # Write the length and the serialized data to the stream
    stream.write(length + pickled_data)
    # Ensure all data is sent
    await stream.drain()
