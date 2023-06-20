"""Process for receiving frames from the optical tracker and pushing them to a queue."""

import logging
import multiprocessing as mp
import time
from typing import Set, Tuple
import numpy as np
import zmq
from scipy.spatial.transform import Rotation as R
from stringcase import snakecase

log = logging.getLogger(__name__)


def from_quatpos(qp: np.ndarray) -> np.ndarray:
    """Convert a quaternion and position to a 4x4 transform.

    Args:
        qp: A (7) array containing the quaternion and position.

    Returns:
        A 4x4 transform matrix.
    """
    q = qp[:4]
    p = qp[4:]
    r = R.from_quat(q).as_matrix()
    t = np.eye(4)
    t[:3, :3] = r
    t[:3, 3] = p
    return t


class FrameListener(mp.Process):
    # to_frame, from_frame, filename

    def __init__(self, corridor_server_queue: mp.Queue, ip: str, port: int) -> None:
        """Frame listener.

        Args:
            corridor_server_queue (mp.Queue): Queue for sending data to the corridor server.
            ip (str): IP address of the plus-tracker server.
            port (int): Port of the plus-tracker server."""
        super().__init__()
        self.corridor_server_queue = corridor_server_queue
        self.ip = ip
        self.port = port

    def run(self):
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.connect(f"tcp://{self.ip}:{self.port}")
        socket.setsockopt(zmq.RCVTIMEO, 5000)
        socket.setsockopt_string(zmq.SUBSCRIBE, "")

        # Startup delay to let other programs get setup.
        time.sleep(1)

        while True:
            try:
                msg = socket.recv_multipart()
                message_type = msg[0].decode("utf-8")
                if message_type == "frame":
                    to_frame = snakecase(msg[1].decode("utf-8"))
                    from_frame = snakecase(msg[2].decode("utf-8"))
                    transform = from_quatpos(np.frombuffer(msg[3], dtype="<f"))
                    timestamp = msg[4].decode("utf-8")
                    self.corridor_server_queue.put(
                        ("tracker_frame", to_frame, from_frame, transform, timestamp)
                    )
                else:
                    log.warning(f"Unknown message type {message_type}")
            except KeyboardInterrupt:
                log.info("\nExiting...")
                exit(1)
            except Exception as e:
                time.sleep(0.1)
                continue
