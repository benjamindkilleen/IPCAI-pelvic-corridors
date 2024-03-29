from typing import Tuple
from numpy.lib.function_base import disp
import socket
import numpy as np
import scipy.io as io
from scipy.spatial.transform import Rotation


flipx = np.array(
    [
        [1, -1, -1, -1],
        [-1, 1, 1, 1],
        [-1, 1, 1, 1],
        [-1, 1, 1, 1],
    ]
)

flipy = np.array(
    [
        [1, -1, 1, 1],
        [-1, 1, -1, -1],
        [1, -1, 1, 1],
        [1, -1, 1, 1],
    ]
)

flipz = np.array(
    [
        [1, 1, -1, 1],
        [1, 1, -1, 1],
        [-1, -1, 1, -1],
        [1, 1, -1, 1],
    ]
)


def convert_from_unity_transform(transform: np.ndarray) -> np.ndarray:
    """Converts a transform from Unity to numpy.

    Args:
        transform (np.ndarray): The transform to convert.

    Returns:
        np.ndarray: The converted transform.

    """
    transform = transform * flipz
    transform[:3, 3] = transform[:3, 3] * 1000
    return transform


def convert_to_unity_transform(transform: np.ndarray) -> np.ndarray:
    """Converts a transform from numpy to Unity.

    Args:
        transform (np.ndarray): The transform to convert.

    Returns:
        np.ndarray: The converted transform.

    """
    transform[:3, 3] = transform[:3, 3] / 1000
    transform = transform * flipz
    return transform


def to_qp(transform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert from 4x4 matrix to quaternion and position of the frame.

    Args:
        transform (np.ndarray): The transform to convert.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The quaternion and position.

    """
    q = Rotation.from_matrix(transform[:3, :3]).as_quat()
    p = transform[:3, 3]
    return q, p


class AHATRawFrame:
    def __init__(self, _RawDepth, _RawReflectivity, _timestamp):
        self.RawDepth = _RawDepth
        self.RawReflectivity = _RawReflectivity
        self.timestamp = _timestamp


class AHATFrame:
    def __init__(
        self,
        _MatDepthProcessed,
        _MatReflectivityProcessed,
        _MatDepth,
        _MatReflectivity,
        _timestamp,
        _pose,
    ):
        self.MatDepth = _MatDepth
        self.MatReflectivity = _MatReflectivity
        self.MatDepthProcessed = _MatDepthProcessed
        self.MatReflectivityProcessed = _MatReflectivityProcessed
        self.timestamp = _timestamp
        self.pose = _pose


class VLCRawFrame:
    def __init__(self, _RawVLC, _timestamp):
        self.RawVLC = _RawVLC
        self.timestamp = _timestamp


class VLCFrame:
    def __init__(self, _MatVLC, _MatVLCOrigin, _timestamp):
        self.MatVLC = _MatVLC
        self.MatOrigin = _MatVLCOrigin
        self.timestamp = _timestamp


class SensorType:
    AHAT_CAMERA = 4
    LEFT_FRONT = 0
    RIGHT_FRONT = 1
    LEFT_LEFT = 2
    RIGHT_RIGHT = 3


class Sensor_Network:
    def __init__(self, port, sensortype):
        self._SOCKET = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._PORT = port
        self._TYPE = sensortype
        self._ADDR = ("", port)
        self._CONNECTED = False

    def connect(self):
        self._SOCKET.bind(self._ADDR)
        self._SOCKET.listen(1)
        self._CONN, addr = self._SOCKET.accept()
        self._CONNECTED = True
        return addr

    def _send(self, mess):
        self._CONN.send(mess.encode("UTF-8"))

    def require(self):
        if self._TYPE == SensorType.AHAT_CAMERA:
            mess = "Recv image"
            self._send(mess)
            Ab_data_raw = b""
            Depth_data_raw = b""
            Time_data = b""
            len_tm = 32
            while len_tm:
                buf = self._CONN.recv(len_tm)
                if not buf:
                    print("ERROR when recving")
                    return None
                Time_data += buf
                len_tm -= len(buf)

            ## 52488=512*512*2
            len_ = 524288
            while len_:
                buf = self._CONN.recv(len_)
                if not buf:
                    print("ERROR when recving")
                    return None
                Ab_data_raw += buf
                len_ -= len(buf)

            len_ = 524288
            while len_:
                buf = self._CONN.recv(len_)
                if not buf:
                    print("ERROR when recving")
                    return None
                Depth_data_raw += buf
                len_ -= len(buf)
            # print(Time_data)
            return AHATRawFrame(Depth_data_raw, Ab_data_raw, int(Time_data.decode("UTF8")))
            # return (Ab_data_raw, Depth_data_raw, Time_data)
        else:
            len_ = 640 * 480
            mess = "Recv image"

            self._send(mess)
            # print("Req")
            img_raw = b""
            Time_data = b""
            len_tm = 32
            while len_tm:
                buf = self._CONN.recv(len_tm)
                if not buf:
                    print("ERROR when recving")
                    return None
                Time_data += buf
                len_tm -= len(buf)

            while len_:
                buf = self._CONN.recv(len_)
                if not buf:
                    print("ERROR when recving")
                    return None
                img_raw += buf
                len_ -= len(buf)
                # print(len_)
            # print(int(Time_data.decode("UTF-8")))
            return VLCRawFrame(img_raw, int(Time_data.decode("UTF-8")))
            # return (img_raw,Time_data)


class SimpleFPSCalculator:
    def __init__(self):
        self.FrameTime = []

    def framerate(self, _tm):
        self.FrameTime.append(_tm)
        if len(self.FrameTime) > 40:
            self.FrameTime.pop(0)
            return int(len(self.FrameTime) / (self.FrameTime[-1] - self.FrameTime[0]))
        else:
            return -1
