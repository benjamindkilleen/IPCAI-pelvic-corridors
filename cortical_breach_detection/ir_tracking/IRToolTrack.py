import logging
from pathlib import Path
import numpy as np
import cv2
import os
import json
from scipy.io import loadmat
import queue
from typing import List, Dict
import copy

from ..utils.rom_utils import parse_rom_file
from ..utils.track_utils import AHATFrame
from .AHATNDITracker import NDI_Tool
from .ThreadFactory import AHATIRToolTracking

log = logging.getLogger(__name__)


class IRToolTrack:
    def __init__(self, tool_dir: str, tool_names: List[str]):
        ## Variant for sensor datas
        self.down_limit_ab = 128
        self.up_limit_ab = 512
        self._mx = 5

        # TODO: make this an mp.Queue.
        self.AHATForCalFrame = queue.Queue(maxsize=self._mx)
        self.AHAT_isworking = False

        self.tool_dir = Path(tool_dir).expanduser()
        self.tool_names = tool_names

        self.tools = self.load_tools()

    def track_tool(self, lut, radius):
        self._isTrackingTool = True

        self.tracking_ir_thread = AHATIRToolTracking(
            self.AHATForCalFrame,
            self.tools,
            self.tool_names,
            lut,
            "12349",
            radius,
        )
        self.tracking_ir_thread.start()

    def add_frame(self, frame_depth, frame_ab, pose, timestamp):
        if not self.AHATForCalFrame.full():
            Ab_low = self.down_limit_ab
            Ab_high = self.up_limit_ab

            _, imgdepth = cv2.threshold(frame_depth, 1000, 0, cv2.THRESH_TRUNC)
            imgdepth_processed = np.uint8(imgdepth / 4)
            imgab_processed = copy.deepcopy(frame_ab)
            imgab_processed[imgab_processed < Ab_low] = Ab_low
            imgab_processed[imgab_processed > Ab_high] = Ab_high
            imgab_processed = np.uint8((imgab_processed - Ab_low) / (Ab_high - Ab_low) * 255)
            Frame_this = AHATFrame(
                imgdepth_processed,
                imgab_processed,
                frame_depth,
                frame_ab,
                timestamp,
                pose,
            )

            self.AHATForCalFrame.put(Frame_this)
        else:
            print(".", end="", flush=True)

    def get_poses(self) -> Dict[str, np.ndarray]:
        if self.tracking_ir_thread.is_alive():
            return self.tracking_ir_thread.get_poses()
        else:
            return dict()

    def load_tools(self):
        tool_paths = self.tool_dir.glob("*.rom")
        available_tool_names = set(p.name for p in tool_paths)
        tools = []
        for i, name in enumerate(self.tool_names):
            if f"{name}.mat" in available_tool_names:
                path = self.tool_dir / f"{name}.mat"
                tool = loadmat(path)
                points = np.array(tool["ToolShape"])
                tools.append(NDI_Tool(tool["ToolShape"]))
            elif f"{name}.rom" in available_tool_names:
                path = self.tool_dir / f"{name}.rom"
                points = parse_rom_file(path)  # list of points
                points = np.array(points)
                tools.append(NDI_Tool(points))
            else:
                raise Exception("Tool Cannot Find : " + name)

            # print the unity coordinates for convenience.
            points_lhs = points.copy()
            points_lhs[:, 2] *= -1
            points_lhs /= 1000.0
            log.info(f"{self.tool_names[i]} spheres in unity coords:\n{points_lhs}")

        return tools
