import asyncio
import socket
import sys

from .comms import _receive_pickled_data, _send_pickled_data

BLENDER_EXE_PATH = "/blender/blender"
BLENDER_SCRIPT_PATH = "/modules/blendervsim/blenderscripts/render_main.py"

import socket
import struct
import subprocess
import pickle
import threading


class BlenderVSim(object):
    def __init__(
        self,
        blender_exe_path=BLENDER_EXE_PATH,
        blender_script_path=BLENDER_SCRIPT_PATH,
        blender_scene_path=None,
        verbose=False,
        debug=False,
    ):
        self.blender_exe_path = blender_exe_path
        self.blender_script_path = blender_script_path
        self.blender_scene_path = blender_scene_path
        self.blender_comm_port = None
        self.verbose = verbose
        self.debug = debug
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.blender_process = None
        self.conn = None
        self._stdout_thread = None

    def __enter__(self):
        self._startup()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._shutdown()

    def _startup(self):
        # Set up the socket
        self.server_socket.bind(("localhost", 0))
        self.blender_comm_port = self.server_socket.getsockname()[1]
        self.server_socket.listen(1)

        # Start blender
        self._start_blender_subprocess()

        # Listen on stdout
        self._stdout_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._stdout_thread.start()
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._stderr_thread.start()
        self._error_logs = ""

        # Listen for data from Blender
        self.conn, _ = self.server_socket.accept()

    def _start_blender_subprocess(self):
        blender_command = [
            self.blender_exe_path,
            "--background",
            "--python",
            self.blender_script_path,
            "--",
            "--comm-port",
            str(self.blender_comm_port),
        ]
        if self.blender_scene_path:
            blender_command.insert(1, self.blender_scene_path)

        stderr_pipe = sys.stderr if self.debug else subprocess.PIPE

        self.blender_process = subprocess.Popen(
            blender_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr_pipe,
            text=False,  # Binary mode
        )

    def _shutdown(self):
        """Cleanly shuts down the Blender subprocess and closes connections."""
        try:
            # Send a close command to Blender
            _send_pickled_data(self.blender_process.stdin, {"command": "close"})
            self.conn.close()
            self.blender_process.stdin.close()
        except (ConnectionResetError, BrokenPipeError):
            pass

        if self.blender_process:
            self.blender_process.wait()

        if self.server_socket:
            self.server_socket.close()

    def _read_stdout(self):
        """Reads Blender's stdout in a separate thread."""
        while True:
            line = self.blender_process.stdout.readline()
            if not line:
                break
            if self.verbose:
                print(f"[Blender Log] {line.decode().rstrip()}")

    def _read_stderr(self):
        """Reads Blender's stderr in a separate thread."""
        while True:
            line = self.blender_process.stderr.readline()
            if not line:
                if not self.conn:
                    # If blender never connected, reply on its behalf
                    tmp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    tmp_sock.connect(("localhost", self.blender_comm_port))
                    tmp_sock.close()
                break
            self._error_logs += f"[Blender Error] {line.decode()}"

    def _send_receive_data(self, data):
        """Sends data to Blender and waits for the response."""
        try:
            _send_pickled_data(self.blender_process.stdin, data)
            data = _receive_pickled_data(self.conn)
            if data is None:
                raise BrokenPipeError(f"No data returned from Blender.")
            return data
        except (ConnectionResetError, BrokenPipeError) as e:
            # Finish reading from stdout and stderr, then raise error
            self._stdout_thread.join()
            self._stderr_thread.join()
            raise RuntimeError(
                f"Blender has died with the following error:\n\n{self._error_logs}"
            )

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
                out_data = self._send_receive_data(
                    {"command": name, "args": args, "kwargs": kwargs}
                )
                return out_data["output"]

            return dynamic_method
