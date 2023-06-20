"""Server for communicating with the hololens."""
import logging
import multiprocessing as mp
import struct
from typing import Any, Dict
import zmq
import cv2
import time
import numpy as np
from deepdrr import geo
import traceback
from queue import Empty

from ..ir_tracking.IRToolTrack import IRToolTrack
from ..threads import LUTListener
from ..utils.track_utils import convert_from_unity_transform

log = logging.getLogger(__name__)

# If the frame name differs from tool definition filename.
TOOL_NAME_MAPPING = {
    "brainlab_pointer": "pointer",
    "brainlab_reference": "reference",
}


class HololensServer(mp.Process):
    def __init__(
        self,
        queue: mp.Queue,
        corridor_server_queue: mp.Queue,
        ip: int,
        port: int,
        sphere_radius: float,
        lut_listener: Dict[str, Any] = {},
        ir_tool_track: Dict[str, Any] = {},
    ):
        """Hololens server.

        Args:
            queue (mp.Queue): Queue to receive data from the corridor server.
            corridor_server_queue (mp.Queue): Queue to send data to the corridor server.
            ip (int): IP address of the hololens.
            port (int): Port of the hololens.
            sphere_radius (float): Radius of the sphere to use for tracking.
            lut_listener (Dict[str, Any], optional): Configuration for the LUT listener. Defaults to {}.
            ir_tool_track (Dict[str, Any], optional): Configuration for the IR tool tracker. Defaults to {}.
        """
        super().__init__()
        self.queue = queue
        self.corridor_server_queue = corridor_server_queue
        self.ip = ip
        self.port = port
        self.sphere_radius = sphere_radius
        self.lut_listener_config = lut_listener
        self.ir_tool_track_config = ir_tool_track
        self.tool_tracker = None
        self.lut_listener = None
        self.tracking = True

    def run(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.subscribe("")
        self.socket.connect(f"tcp://{self.ip}:{self.port}")
        log.info(f"Subscriber connected to HoloLens at {self.ip}:{self.port}")

        self.lut_listener = LUTListener(**self.lut_listener_config)
        self.lut_listener.start()

        # TODO: smooth the incoming transforms from hololens.

        lut = None
        t0 = time.time()
        t_track = time.time()
        while True:
            try:
                if time.time() - t0 > 5 and self.tool_tracker is None:
                    log.info('HoloLens Server waiting... (Toggle "IR Tracking" on left wrist?)')
                    t0 = time.time()

                lut = self.lut_listener.get_lut()
                if lut is not None:
                    log.info("LUT received.")
                    if self.tool_tracker is not None:
                        # Restart the tracking thread.
                        self.tool_tracker.tracking_ir_thread.stop()
                        self.tool_tracker = None

                    # TODO: make this a process, not a thread.
                    self.tool_tracker = IRToolTrack(**self.ir_tool_track_config)
                    self.tool_tracker.track_tool(lut, self.sphere_radius)

                if self.tool_tracker is None:
                    time.sleep(0.1)
                    continue

                try:
                    item = self.queue.get(block=False)
                except Empty:
                    item = None

                if item is not None:
                    msg = item[0]
                    if msg == "toggle":
                        self.tracking = not self.tracking
                        log.info(f"Tracking: {self.tracking}")
                    else:
                        log.error(f"Unknown message: {msg}")

                if not self.tracking:
                    time.sleep(0.1)
                    continue

                msg = self.socket.recv_multipart()
                message_type = msg[0].decode("utf-8")
                if message_type == "depth_frame":
                    # log.info("Received depth frame.")
                    timestamp = struct.unpack("q", msg[1])[0]  # ulong

                    cam_pose_array = np.frombuffer(msg[2], np.dtype(np.float32))
                    pose = np.reshape(cam_pose_array, (4, 3)).T

                    frame_depth = np.frombuffer(msg[3], np.uint16).reshape((512, 512))
                    frame_ab = np.frombuffer(msg[4], np.uint16).reshape((512, 512))

                    self.tool_tracker.add_frame(frame_depth, frame_ab, pose, timestamp)

                    # Get the poses (requires a lock)
                    # TODO: make this a queue from the tracker thread (which shoudld be a process)
                    tool_poses = self.tool_tracker.get_poses()

                    for tool_name, transform in tool_poses.items():
                        if tool_name in TOOL_NAME_MAPPING:
                            tool_name = TOOL_NAME_MAPPING[tool_name]
                        self.corridor_server_queue.put(
                            ("hololens_frame", "hlworld", tool_name, transform)
                        )

                elif message_type == "handle":
                    p = np.frombuffer(msg[1], np.dtype(np.float32))
                    q = np.frombuffer(msg[2], np.dtype(np.float32))
                    world_from_handle_lhs = geo.FrameTransform.from_rt(q, p).data
                    world_from_handle = convert_from_unity_transform(world_from_handle_lhs)
                    self.corridor_server_queue.put(
                        ("hololens_frame", "hlworld", "handle", world_from_handle)
                    )
                else:
                    log.error(f"Unknown message type: {message_type}")
            except KeyboardInterrupt:
                log.info("Keyboard interrupt received.")
                break
            except zmq.ZMQError as e:
                log.error(f"ZMQ error: {e}")
                continue
            except Exception as e:
                # log.error(f"Exception: {e.with_traceback()}")
                traceback.print_exc()
                continue

    def stop(self):
        if self.lut_listener is not None:
            self.lut_listener.close()
        if self.tool_tracker is not None:
            self.tool_tracker.tracking_ir_thread.stop()
            del self.tool_tracker
        if hasattr(self, "socket"):
            self.socket.close()
        if hasattr(self, "context"):
            self.context.term()
