import logging
from queue import Empty
import threading
from typing import Optional
import numpy as np
import zmq
import time
import multiprocessing as mp

log = logging.getLogger(__name__)

from ..utils.track_utils import convert_to_unity_transform, to_qp


class PosePublisher(threading.Thread):
    """
    Thread that publishes component poses to the HoloLens.

    Give it the 4x4 matrix in DeepDRR space. Handles conversion to Unity space.
    """

    def __init__(self, pose_queue: mp.Queue, port: int = 12351) -> None:
        super().__init__()
        self.port = port
        self.pose_queue = pose_queue

    def run(self):
        """Runs forever, waiting for image updates from the server."""
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.set_hwm(5)
        self.socket.bind(f"tcp://*:{self.port}")
        log.info(f"ComponentPublisher bound to port {self.port}")

        while True:
            try:
                item = self.pose_queue.get(block=True, timeout=0.1)
                idx = item[0]
                transform = item[1]
                if len(item) > 2:
                    scale = item[2]
                else:
                    scale = np.array([1, 1, 1])
                transform_unity = convert_to_unity_transform(transform)
                q, p = to_qp(transform_unity)
                q_msg = q.astype("<f").tobytes()
                p_msg = p.astype("<f").tobytes()
                scale_msg = scale.astype("<f").tobytes()
                idx_msg = np.array(idx).astype("<i").tobytes()
                # TODO: incorporate scale into the message, requires building a new listener, probably.
                self.socket.send_multipart([b"ir_tool", p_msg, q_msg, idx_msg])
            except Empty:
                pass

            except KeyboardInterrupt:
                break

            except Exception as e:
                log.exception(e)

    def close(self):
        """Close the socket."""
        self.socket.close()
        self.context.term()
