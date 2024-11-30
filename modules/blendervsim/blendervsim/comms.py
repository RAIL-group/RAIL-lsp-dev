import asyncio
import struct
import socket
import pickle


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

def _send_pickled_data(stream, data):
    """Serialize the data with pickle and send it over an asyncio stream."""
    # Serialize the data
    pickled_data = pickle.dumps(data)
    # Prepend the length of the serialized data (4 bytes, big-endian)
    length = struct.pack('>I', len(pickled_data))
    # Write the length and the serialized data to the stream
    stream.write(length + pickled_data)
    # Ensure all data is sent
    stream.flush()
