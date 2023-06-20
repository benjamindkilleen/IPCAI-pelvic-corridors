from datetime import datetime
from pathlib import Path
from typing import List

# Only import one
import autograd.numpy as np

# import numpy as np

import click
import cv2
from deepdrr.utils import data_utils, dicom_utils, image_utils
import logging
from rich.logging import RichHandler
import matplotlib.pyplot as plt
from rich.progress import track
import pyvista as pv

# from pytransform3d import rotations as pr
# from pytransform3d import transformations as pt
# from pytransform3d.transform_manager import TransformManager

from cortical_breach_detection.loopx import DEGREE_SIGN
import cortical_breach_detection.viewpoint_planning as vpp


logging.basicConfig(
    level="DEBUG",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger(__name__)

l = np.array(
    [
        [301.52, -501.57, 117.59],
        [158.36, -439.77, 110.92],
        [5.82, 594.93, -166.36],
        [-79.58, 495.47, -132.51],
        [-70.46, -230.06, 69.61],
        [-127.72, -137.72, 47.83],
        [-187.94, 218.71, -47.08],
    ]
)


def fromstring(s: str) -> np.ndarray:
    """Convert a string representation of a 4x4 matrix into a numpy array."""
    return np.array(s.split()).astype(float).reshape(4, 4)


def inverse(F: np.ndarray) -> np.ndarray:
    """Inverse of a 4x4 transformation matrix"""
    R = F[:3, :3]
    t = F[:3, 3]
    return np.vstack((np.hstack((R.T, -R.T @ t[:, np.newaxis])), np.array([0, 0, 0, 1])))


def skew(v):
    """Skew symmetric matrix.

    Args:
        v (np.array): (3,) array.

    Returns:
        np.array: (3, 3) skew symmetric matrix.
    """
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def from_rotvec(r):
    """Convert rotation vector to rotation matrix using Rodrigues' formula.

    Args:
        r (np.array): (3,) array of rotation vector.

    Returns:
        np.array: (3, 3) rotation matrix.
    """
    theta = np.linalg.norm(r)
    if theta == 0:
        return np.eye(3)
    r = r / theta
    return np.eye(3) + np.sin(theta) * skew(r) + (1 - np.cos(theta)) * skew(r) @ skew(r)


def fromquat(q):
    """Convert quaternion to rotation matrix.

    Args:
        q (np.array): (4,) array of quaternion [x, y, z, w].

    Returns:
        np.array: (3, 3) rotation matrix.
    """
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x**2 - 2 * y**2],
        ]
    )


def from_quatpos(quatpos):
    """Convert quaternion and position to transformation matrix.

    Args:
        quatpos (np.array): (7,) array of quaternion and position [x, y, z, w, px, py, pz].

    Returns:
        np.array: (4, 4) transformation matrix.
    """
    q = quatpos[:4]
    p = quatpos[4:]
    return np.vstack((np.hstack((fromquat(q), p[:, np.newaxis])), np.array([0, 0, 0, 1])))


def loopx_objective(
    x: np.ndarray,
    spheres_in_index: np.ndarray,
    index_from_ring: np.ndarray,
    tracker_from_loopx: np.ndarray,
    tracker_from_pointer: np.ndarray,
    spheres_in_pointer: np.ndarray,
    pixel_size: np.ndarray,
    verbose: bool = False,
    image_paths: List[Path] = None,
) -> float:
    """Get the squared error between estimated and actual image positions.

    Let's say we take N images

    Args:
        x (np.array): (7,) array giving the quaternion and translation of the ring_from_loopx transform.
        spheres_in_index (np.array): (N, M, 2) array of homogeneous image positions (col, row) for each of the M spheres.
        index_from_ring (np.array): (N, 3, 4) projection matrix.
        tracker_from_loopx (np.array): (N, 4, 4) array of poses of the LoopX in the tracker frame.
        tracker_from_pointer (np.array): (N, 4, 4) array of poses of the pointer in the tracker frame.
        spheres_in_pointer (np.array): (M, 4) array containing the homogeneous coordinates of the M spheres in the pointer frame.
        pixel_size (float): (N, 2) Size of a (row, col) pixel in the image, in mm.

    Returns:
        float: MSE between estimated and actual image positions.
    """
    if verbose:
        log.debug(f"x: {x.shape}")
        log.debug(f"spheres_in_index: {spheres_in_index.shape}")
        log.debug(f"index_from_isocenter: {index_from_ring.shape}")
        log.debug(f"tracker_from_loopx: {tracker_from_loopx.shape}")
        log.debug(f"tracker_from_pointer: {tracker_from_pointer.shape}")
        log.debug(f"spheres_in_pointer: {spheres_in_pointer.shape}")
        log.debug(f"pixel_size: {pixel_size.shape}")

    # Unpack optimization parameters
    ring_from_loopx = from_quatpos(x[0:7])

    # TODO; This is for debugging to only do the gantry tilt.
    u = spheres_in_index
    N = spheres_in_index.shape[0]
    M = spheres_in_index.shape[1]
    e = np.array(0.0)
    derr = np.array(0.0)
    for n in range(N):
        # rdir = row_direction_in_ring[n, :3]
        # cdir = col_direction_in_ring[n, :3]
        # detector_normal = np.cross(rdir, cdir)
        # detector_normal = np.concatenate((detector_normal, [0]))
        # detector_plane_in_ring = vpp.plane_from_point_normal(
        # detector_origin_in_ring[n], detector_normal
        # )
        ring_from_pointer = (
            ring_from_loopx @ inverse(tracker_from_loopx[n]) @ tracker_from_pointer[n]
        )
        u_hat = []
        for m in range(M):
            # Get the sphere position in the ring frame
            sphere_in_ring = ring_from_pointer @ spheres_in_pointer[m]
            u_hat_m = index_from_ring[n] @ sphere_in_ring
            u_hat_m = u_hat_m / u_hat_m[2]

            # l = vpp.join(
            #     source_in_ring[n],
            #     sphere_in_ring,
            # )

            # sphere_on_detector = vpp.meet(l, detector_plane_in_ring)
            # sphere_on_detector = sphere_on_detector / sphere_on_detector[3]

            # v = sphere_on_detector - detector_origin_in_ring[n]
            # x = np.dot(v, col_direction_in_ring[n]) / pixel_size[n, 1]
            # y = np.dot(v, row_direction_in_ring[n]) / pixel_size[n, 0]
            u_hat.append(u_hat_m)

            # Compute the error
            e_ = (u_hat_m[0] - u[n, m, 0]) ** 2 + (u_hat_m[1] - u[n, m, 1]) ** 2
            e += e_
            derr_ = np.sqrt(e_)
            if verbose:
                log.info(f"Image {n} marker {m} derr: {derr_}")
            derr += derr_

        u_hat = np.array(u_hat)

        if verbose and image_paths is not None:
            # log.info(f"u_hat: {u_hat},\nu: {u[n]},\ne: {np.linalg.norm(u_hat - u[n], axis=1)}")
            image = image_utils.ensure_cdim(
                image_utils.as_uint8(dicom_utils.read_image(image_paths[n]))
            )
            for j in range(u_hat.shape[0]):
                cv2.circle(image, (int(u[n, j, 0]), int(u[n, j, 1])), 25, (0, 0, 255), 3)
                cv2.circle(image, (int(u_hat[j, 0]), int(u_hat[j, 1])), 25, (255, 0, 0), 3)

            plot_path = f"images/loopx_cal_{image_paths[n].stem}.png"
            image_utils.save(plot_path, image)
            log.info(f"Saved image to {plot_path}")

    e = e / N / M
    derr = derr / N / M

    log.debug(f"mean distance error: {derr}")
    return e
