import asyncio
import socket
import sys

from .comms import _receive_pickled_data, _send_pickled_data

BLENDER_EXE_PATH = '/blender/blender'
BLENDER_SCRIPT_PATH = '/modules/blendervsim/blenderscripts/render_gridmap_figure.py'

class BlenderVSim(object):
    def __init__(self, blender_exe_path=BLENDER_EXE_PATH,
                 blender_script_path=BLENDER_SCRIPT_PATH,
                 blender_scene_path=None,
                 verbose=False,
                 debug=False):
        self.blender_exe_path = blender_exe_path
        self.blender_script_path = blender_script_path
        self.blender_scene_path = blender_scene_path
        self.blender_comm_port = None
        self.verbose = verbose
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.loop = asyncio.new_event_loop()
        self._stdout_task = None
        self._stderr_task = None
        self._stderr_logs = ''
        self.debug = debug

    def __enter__(self):
        asyncio.set_event_loop(self.loop)
        return self.loop.run_until_complete(self._async_enter())

    def __exit__(self, exc_type, exc_value, traceback):
        self.loop.run_until_complete(self._async_exit())
        self.loop.close()

    async def _create_blender_subprocess(self):
        if self.blender_scene_path:
            blender_command = [self.blender_exe_path, self.blender_scene_path]
        else:
            blender_command = [self.blender_exe_path]

        if self.debug:
            stderr_pipe = sys.stderr
        else:
            stderr_pipe = asyncio.subprocess.PIPE

        self.blender_process = await asyncio.create_subprocess_exec(
            *blender_command, '--background', '--python', self.blender_script_path,
            '--', '--comm-port', str(self.blender_comm_port),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=stderr_pipe
        )

    async def _async_enter(self):
        self.server_socket.bind(("localhost", 0))
        self.blender_comm_port = self.server_socket.getsockname()[1]
        self.server_socket.listen(1)

        await self._create_blender_subprocess()

        loop = asyncio.get_event_loop()
        self.conn, self.addr = await loop.run_in_executor(None, self.server_socket.accept)

        # Start listening to stdout asynchronously
        self._stdout_task = loop.create_task(self._read_stdout())

        return self

    async def _async_exit(self):
        # Cancel stdout and stderr listeners
        if self._stdout_task:
            self._stdout_task.cancel()

        # Send a message to blender to close
        try:
            await _send_pickled_data(self.blender_process.stdin, {'command': 'close'})
            self.blender_process.stdin.close()
        except (ConnectionResetError, BrokenPipeError) as e:
            pass
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
            self._stderr_logs += f"[Blender Error] {line.decode().rstrip()}\n"

    def _send_receive_data(self, data):
        """Synchronous interface for sending and receiving data."""
        try:
            return self.loop.run_until_complete(self._async_send_receive_data(data))
        except (ConnectionResetError, BrokenPipeError) as e:
            self.loop.run_until_complete(self._read_stderr())
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
            raise BrokenPipeError(f"The connection to Blender is unexpectedly closed.")

        return received_data

    def __getattr__(self, name):
        """
        Handle method calls dynamically. If a method exists, call it.
        Otherwise, handle it as an arbitrary function.
        """
        # Use __getattribute__ to safely check for existing attributes/methods
        try:
            attr = object.__getattribute__(self, name)
            if callable(attr):
                return attr  # Return the existing callable
            else:
                raise AttributeError(f"'{name}' exists but is not callable.")
        except AttributeError:
            # If the attribute does not exist, create a dynamic method
            def dynamic_method(*args, **kwargs):
                out_data = self._send_receive_data({'command': name,
                                                    'args': args,
                                                    'kwargs': kwargs})
                return out_data['output']
            return dynamic_method


import socket
import struct
import subprocess
import pickle
import threading
import sys

class BlenderVSim(object):
    def __init__(self, blender_exe_path=BLENDER_EXE_PATH,
                 blender_script_path=BLENDER_SCRIPT_PATH,
                 blender_scene_path=None,
                 verbose=False,
                 debug=False):
        self.blender_exe_path = blender_exe_path
        self.blender_script_path = blender_script_path
        self.blender_scene_path = blender_scene_path
        self.blender_comm_port = None
        self.verbose = verbose
        self.debug = debug
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.blender_process = None
        self.conn = None
        self.addr = None
        self._stdout_thread = None
        self._stderr_thread = None
        self._stderr_logs = ''

    def __enter__(self):
        self._setup_socket()
        self._start_blender_subprocess()
        self._accept_connection()
        self._start_stdout_listener()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._shutdown()

    def _setup_socket(self):
        self.server_socket.bind(("localhost", 0))
        self.blender_comm_port = self.server_socket.getsockname()[1]
        self.server_socket.listen(1)

    def _start_blender_subprocess(self):
        blender_command = [self.blender_exe_path, '--background', '--python', self.blender_script_path,
                           '--', '--comm-port', str(self.blender_comm_port)]
        if self.blender_scene_path:
            blender_command.insert(1, self.blender_scene_path)

        stderr_pipe = sys.stderr if self.debug else subprocess.PIPE

        self.blender_process = subprocess.Popen(
            blender_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr_pipe,
            text=False  # Binary mode
        )

    def _accept_connection(self):
        self.conn, self.addr = self.server_socket.accept()

    def _start_stdout_listener(self):
        self._stdout_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._stdout_thread.start()

    def _read_stdout(self):
        """Reads Blender's stdout in a separate thread."""
        while True:
            line = self.blender_process.stdout.readline()
            if not line:
                break
            if self.verbose:
                print(f"[Blender Log] {line.decode().strip()}")

    def _read_stderr(self):
        """Reads Blender's stderr in a separate thread."""
        while True:
            line = self.blender_process.stderr.readline()
            if not line:
                break
            self._stderr_logs += f"[Blender Error] {line.decode()}"

    def send_receive_data(self, data):
        """Sends data to Blender and waits for the response."""
        try:
            self._send_pickled_data(data)
            data = _receive_pickled_data(self.conn)
            if data is None:
                raise BrokenPipeError(f"No data returned from Blender.")
            return data
        except (ConnectionResetError, BrokenPipeError) as e:
            error_msg = self._read_remaining_stderr()
            raise RuntimeError(f"Blender has died with the following error: {e}\n\n{error_msg}")

    def _send_pickled_data(self, data):
        """Pickle and send data to Blender."""
        pickled_data = pickle.dumps(data)
        # Prepend the length of the serialized data (4 bytes, big-endian)
        length = struct.pack('>I', len(pickled_data))
        # Write the length and the serialized data to the stream
        self.blender_process.stdin.write(length + pickled_data)
        self.blender_process.stdin.flush()

    def _read_remaining_stderr(self):
        """Read remaining stderr logs."""
        stderr_logs = ''
        if not self.debug and self.blender_process.stderr:
            for line in self.blender_process.stderr:
                stderr_logs += f"[Blender Error] {line.decode()}"

        return stderr_logs

    def _shutdown(self):
        """Cleanly shuts down the Blender subprocess and closes connections."""
        try:
            # Send a close command to Blender
            self._send_pickled_data({'command': 'close'})
            self.conn.close()
            self.blender_process.stdin.close()
        except (ConnectionResetError, BrokenPipeError):
            pass

        if self.blender_process:
            self.blender_process.wait()

        if self.server_socket:
            self.server_socket.close()

    def __getattr__(self, name):
        """
        Handle method calls dynamically. If a method exists, call it.
        Otherwise, handle it as an arbitrary function.
        """
        # Use __getattribute__ to safely check for existing attributes/methods
        try:
            attr = object.__getattribute__(self, name)
            if callable(attr):
                return attr  # Return the existing callable
            else:
                raise AttributeError(f"'{name}' exists but is not callable.")
        except AttributeError:
            # If the attribute does not exist, create a dynamic method
            def dynamic_method(*args, **kwargs):
                out_data = self.send_receive_data({'command': name,
                                                   'args': args,
                                                   'kwargs': kwargs})
                return out_data['output']
            return dynamic_method
