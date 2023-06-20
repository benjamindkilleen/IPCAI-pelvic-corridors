from typing import List, Optional, Tuple, Dict
import logging
from .. import utils
import os
import copy
import copy
import threading
import zmq
from pathlib import Path
import time
import numpy as np

log = logging.getLogger(__name__)


class LUTListener(threading.Thread):
    """
    Thread that listens for LUT updates from the server.

    The LUT is sent whenever IRTracking is turned on.
    """

    lut: Optional[np.ndarray]

    def __init__(self, port: int = 12348) -> None:
        super().__init__()
        self.port = port
        self.lock = threading.Lock()
        self.lut = None
        self.updated = False

    def get_lut(self) -> Optional[np.ndarray]:
        """Get the LUT, if it is available.

        Returns:
            np.ndarray: The LUT or None.

        """
        with self.lock:
            if self.lut is None or not self.updated:
                lut = None
            else:
                lut = self.lut.copy()
                self.updated = False
        return lut

    def run(self):
        """Runs forever, waiting for LUT updates from the server."""
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.setsockopt(zmq.RCVTIMEO, 5000)
        self.socket.bind(f"tcp://*:{self.port}")
        log.info(f"LUTListener bound to port {self.port}")

        t0 = time.time()
        while True:
            try:
                msg = self.socket.recv_multipart()
                message_type = msg[0].decode("utf-8")

                if message_type != "depth_lut":
                    log.debug("Received string: %s" % message_type)
                    continue

                log.info(f"LUT received. Length: {len(msg[1])}")
                self.socket.send_string("OK")

                with self.lock:
                    self.lut = np.frombuffer(msg[1], np.dtype(np.float32))
                    self.lut = np.reshape(self.lut, (-1, 3))
                    self.updated = True
                log.debug("LUT updated.")
            except KeyboardInterrupt:
                log.info("\nExiting...")
                exit(1)
            except Exception as e:
                if time.time() - t0 > 30:
                    log.info("LUTListener waiting...")
                    t0 = time.time()
                time.sleep(0.1)
                continue

    def close(self):
        self.socket.close()
        self.context.term()
