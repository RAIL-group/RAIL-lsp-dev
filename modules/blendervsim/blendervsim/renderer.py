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


class BlenderVSimOverhead(BlenderVSim):
    def __init__(self, blender_exe_path=BLENDER_EXE_PATH,
                 blender_script_path=BLENDER_SCRIPT_PATH,
                 verbose=False):
        super().__init__(blende_exe_path, blender_script_path, verbose)

        # Now
