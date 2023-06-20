from typing import Dict, List
import numpy as np
import re
from deepdrr import geo
from pathlib import Path
from typing import Tuple
from scipy.spatial.transform import Rotation as R
from stringcase import snakecase
import logging

log = logging.getLogger(__name__)


def fromstring(s: str) -> np.ndarray:
    """Convert a string to a transform.

    Args:
        s: String to convert.

    Returns:
        Transform as a 4x4 numpy array.
    """
    return np.fromstring(s, sep=" ").reshape(4, 4)


def to_quatpos(transform: np.ndarray) -> np.ndarray:
    """Convert a 4x4 transform to a quaternion and position.

    Args:
        transform: 4x4 transform matrix.

    Returns:
        A (7,) array containing the quaternion and position.
    """
    q = R.from_matrix(transform[:3, :3]).as_quat()
    p = transform[:3, 3]
    return np.concatenate((q, p))


def average_transforms(path: Path) -> Dict[Tuple[str, str], geo.F]:
    """Parse the mha file and average the transforms over the whole duration.

    Args:
        path (Path): The path to the mha file.

    Returns:
        Dict[str, geo.F]: Mapping from (to_frame, from_frame) to the average transform over the whole file.
    """
    # Set up the regular expression
    transform_pattern = (
        r"Seq_Frame(?P<frame_idx>\d+)_(?P<from_frame>.*)"
        r"To(?P<to_frame>.*)Transform = (?P<transform>.*)"
    )

    transforms = {}
    transforms: Dict[Tuple[str, str], List[np.ndarray]] = {}
    with open(path, "r") as f:
        lines = f.readlines()

    for line in lines:
        m = re.match(transform_pattern, line)
        if m is None:
            continue
        to_frame = snakecase(m.group("to_frame"))
        from_frame = snakecase(m.group("from_frame"))
        transform = fromstring(m.group("transform"))
        if np.any(np.isnan(transform)):
            continue
        key = (to_frame, from_frame)
        if key in transforms:
            transforms[key].append(to_quatpos(transform))
        else:
            transforms[key] = [to_quatpos(transform)]

    frames = {}
    for (to_frame, from_frame), d in transforms.items():
        qp = np.array(d).mean(0)
        frames[(to_frame, from_frame)] = geo.frame_transform(qp)
    return frames
