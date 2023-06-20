import logging
from queue import Empty
import threading
from typing import Optional
import numpy as np
import zmq
import time
import multiprocessing as mp
import cv2

log = logging.getLogger(__name__)


class ImagePublisher(threading.Thread):
    """
    Thread that publishes images to the server.
    """

    def __init__(self, image_queue: mp.Queue, port: int = 12350) -> None:
        super().__init__()
        self.port = port
        self.queue = image_queue

    def run(self):
        """Runs forever, waiting for image updates from the server."""
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.set_hwm(5)
        self.socket.bind(f"tcp://*:{self.port}")
        log.info(f"ImagePublisher bound to port {self.port}")

        t0 = time.time()
        while True:
            try:
                image = self.queue.get(block=True, timeout=0.1)
                if image.dtype == np.float32:
                    image = (image * 255).astype(np.uint8)
                elif image.dtype == np.uint8:
                    image = image.copy()

                # Image order is flipped on HoloLens. (TODO: figure out why)
                image = np.flip(image, axis=0)
                image_msg = image.tobytes()
                self.socket.send_multipart([b"drr", image_msg])

            except Empty:
                pass

            except KeyboardInterrupt:
                break

            except Exception as e:
                log.exception(e)

    def show_image(self, image: np.ndarray):
        """
        Show the image. Really seems to slow down tracking.
        """
        cv2.imshow("DRR Preview", np.stack([image, image, image], axis=2))
        cv2.waitKey(5)

    def close(self):
        """Close the socket."""
        self.socket.close()
        self.context.term()
