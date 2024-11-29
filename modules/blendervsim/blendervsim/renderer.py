import asyncio
import socket
import subprocess
import sys
import struct
import pickle

BLENDER_EXE_PATH = '/blender/blender'
BLENDER_SCRIPT_PATH = '/modules/blendervsim/blenderscripts/render_gridmap_figure.py'

class BlenderVSim(object):
    def __init__(self, blender_exe_path=BLENDER_EXE_PATH,
                 blender_script_path=BLENDER_SCRIPT_PATH,
                 verbose=False):
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
            # stderr=asyncio.subprocess.PIPE,
        )

        loop = asyncio.get_event_loop()
        self.conn, self.addr = await loop.run_in_executor(None, self.server_socket.accept)

        # Start listening to stdout and stderr asynchronously
        self._stdout_task = loop.create_task(self._read_stdout())
        # self._stderr_task = loop.create_task(self._read_stderr())

        return self

    async def _async_exit(self):
        # Cancel stdout and stderr listeners
        if self._stdout_task:
            self._stdout_task.cancel()
        if self._stderr_task:
            self._stderr_task.cancel()

        # Send a message to blender to close
        try:
            await _send_pickled_data(self.blender_process.stdin, {'command': 'close'})
        except (ConnectionResetError, BrokenPipeError) as e:
            await self.blender_process.wait()
            pass
        self.blender_process.stdin.close()
        await self.blender_process.wait()

    async def _read_stdout(self):
        """Asynchronously read stdout."""
        while True:
            line = await self.blender_process.stdout.readline()
            if not line:
                break
            if self.verbose:
                print(f"[Blender Log] {line.decode().strip()}")  # Replace with custom processing logic

    async def _read_stderr(self):
        """Asynchronously read stderr."""
        while True:
            line = await self.blender_process.stderr.readline()
            if not line:
                break
            print(f"[Blender Error] {line.decode().strip()}")
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
        if not received_data:
            raise RuntimeError(f"Blender has died.")

        return received_data


# Supporting functions for pickle-based communication

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
