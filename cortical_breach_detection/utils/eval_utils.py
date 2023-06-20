"""Functions for interpretting network output."""
from datetime import datetime
import logging
import math
from operator import index
from pathlib import Path
from typing import Optional
from typing import Tuple
import os

import cv2
import numpy as np
from deepdrr import geo
from deepdrr.utils import image_utils
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from scipy.optimize import minimize, differential_evolution


from ..utils import combine_heatmap, imshow_save
from .nn_utils import detect_landmark
from .nn_utils import get_heatmap_threshold
from .. import viewpoint_planning as vp


log = logging.getLogger(__name__)


def locate_landmark(
    heatmap: np.ndarray, hip_mask: Optional[np.ndarray] = None
) -> Optional[geo.Point2D]:
    """Detect a landmark in the heatmap."""

    out = detect_landmark(heatmap, hip_mask=hip_mask)
    if out is None:
        return None
    else:
        return geo.point(out[1], out[0])


def locate_line(
    heatmap: np.ndarray,
    startpoint: geo.Point2D,
    mask: Optional[np.ndarray] = None,
    threshold: Optional[float] = None,
) -> Tuple[Optional[geo.Line2D], Optional[geo.Point2D], Optional[geo.Vector2D]]:
    """Fit a 2D line to the heatmap.

    Args:
        heatmap: The heatmap to fit the line to.
        startpoint: The approximate starting point of the line. The final startpoint returned will
            be the closest point on the line to this point.
        mask: If not None, move the startpoint to the closest point on the line with that is in the mask.

    Returns:

    """
    if threshold is None:
        threshold = get_heatmap_threshold(heatmap, fraction=0.5)
    rr, cc = np.where(heatmap > threshold)
    if len(rr) < 10:
        return None, None, None

    points = np.stack([cc, rr], axis=1)

    # TODO: potential problem when line is vertical.
    model = RANSACRegressor(LinearRegression())
    try:
        model.fit(points[:, 0, np.newaxis], points[:, 1], np.clip(heatmap[rr, cc], 0, None))
    except ValueError:
        return None, None, None
    except ZeroDivisionError:
        return None, None, None

    # y = m x + b -> -m x + y - b = 0
    m = model.estimator_.coef_[0]
    b = model.estimator_.intercept_
    l = geo.line(-m, 1, -b)
    v = l.get_direction()

    # point, direction form of line
    # x = geo.point(*model.params[0])
    # v = geo.vector(*model.params[1])
    # l = geo.line(x, v)
    startpoint_on_line = l.project(startpoint)

    distances = np.linalg.norm(points - np.array(startpoint_on_line)[None, :], axis=1)
    furthest_point = geo.point(
        *points[model.inlier_mask_][np.argmax(distances[model.inlier_mask_])]
    )
    if (furthest_point - startpoint_on_line).dot(v) < 0:
        v = -v

    if mask is not None and mask[startpoint_on_line[1], startpoint_on_line[0]] < 0.5:
        # Move the startpoint to the closest point on the line with that is in the mask.
        points_on_line = np.array(startpoint_on_line) + np.array(v) * np.arange(0, 100)
        points_in_mask = [
            p
            for p in points_on_line
            if p[0] >= 0
            and p[1] >= 0
            and p[0] < mask.shape[1]
            and p[1] < mask.shape[0]
            and mask[int(p[1]), int(p[0])] > 0.5
        ]
        distances = np.linalg.norm(points_in_mask - np.array(startpoint_on_line)[None, :], axis=1)
        startpoint_on_line = geo.point(*points_in_mask[np.argmin(distances)])

    return l, startpoint_on_line, v


def locate_corridor(
    startpoint_heatmap: Optional[np.ndarray],
    corridor_heatmap: np.ndarray,
    mask: Optional[np.ndarray] = None,
    original_image_size: Optional[Tuple[int, int]] = None,
    startpoint: Optional[geo.Point2D] = None,
    move_startpoint_to_mask: bool = False,
    threshold: Optional[float] = None,
) -> Tuple[Optional[geo.Line2D], Optional[geo.Point2D], Optional[geo.Vector2D]]:
    """Locate the corridor and get the right direction.

    Args:
        startpoint_heatmap: The heatmap of the startpoint.
        corridor_heatmap: The heatmap of the corridor.
        original_image_size: The original image size. If given, the output coordinates will be remapped to this indexing. Assumes a simple resize (no change of origin).
        mask: If not None, only fit heatmap values where the mask is > 0.5.
        move_startpoint_in_mask: If True, move the startpoint to the closest point on the line with that is in the mask.

    Returns:
        The line, startpoint and direction of the corridor, in 2D.

    """

    # TODO: use the hip mask to disambiguate the startpoint as well.
    if startpoint is None:
        assert startpoint_heatmap is not None
        startpoint = locate_landmark(startpoint_heatmap)

    if startpoint is None:
        log.info(f"failed to find landmark")
        return None, None, None

    if mask is not None:
        corridor_heatmap = corridor_heatmap.copy()
        corridor_heatmap[mask < 0.5] = 0

    l, x, v = locate_line(
        corridor_heatmap,
        startpoint,
        mask=mask if move_startpoint_to_mask else None,
        threshold=threshold,
    )
    if original_image_size is not None:
        H, W = startpoint_heatmap.shape
        oH, oW = original_image_size
        if oH == oW and H == W:
            index_from_resized = geo.FrameTransform.from_scaling(oH / H, dim=2)
            l = index_from_resized @ l
            x = index_from_resized @ x
            v = index_from_resized @ v
        else:
            raise NotImplementedError("Recommended to resize to square images.")

    return l, x, v


def triangulate_line(
    corridor_in_index_1: geo.Line2D,
    corridor_in_index_2: geo.Line2D,
    index_from_world_1: geo.CameraProjection,
    index_from_world_2: geo.CameraProjection,
) -> geo.Line3D:
    """Triangulate the 3D line from two 2D lines in two images."""
    corridor_plane_1 = corridor_in_index_1.backproject(index_from_world_1)
    corridor_plane_2 = corridor_in_index_2.backproject(index_from_world_2)
    return corridor_plane_1.meet(corridor_plane_2)


def _project_cylinder_edges(
    centerline: geo.Line3D, radius: float, index_from_world: geo.CameraProjection
) -> Tuple[geo.Line2D, geo.Line2D]:
    """Project the edges of the (infinite) cylinder to the image plane."""
    v = centerline.get_direction()
    s = index_from_world.center_in_world  # camera source
    c = centerline.project(s)  # center of cylinder closest to source
    d = c - s  # vector from source to center of cylinder
    theta = np.arcsin(radius / d.norm())  # angle between d and cylinder edge
    p1 = c + d.rotate(v, theta)
    p2 = c + d.rotate(v, -theta)
    g1 = geo.line(p1, p1 + v)
    g2 = geo.line(p2, p2 + v)
    return index_from_world @ g1, index_from_world @ g2


def _project_cylinder_region(
    centerline: geo.Line3D,
    radius: float,
    index_from_world: geo.CameraProjection,
    image_size: Tuple[int, int],
    image: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Project the region of the cylinder to the image plane."""
    g1, g2 = _project_cylinder_edges(centerline, radius, index_from_world)
    cl = index_from_world @ centerline
    mask = np.zeros(image_size, dtype=bool)

    # Get the region between the lines corresponding to the centerline.
    xs, ys = np.meshgrid(np.arange(image_size[1]), np.arange(image_size[0]))
    cp = cl.get_point()

    # Seems to be the slowest part.
    g1_values = xs * g1.a + ys * g1.b + g1.c
    g2_values = xs * g2.a + ys * g2.b + g2.c
    cl_g1_side = cp.x * g1.a + cp.y * g1.b + g1.c
    cl_g2_side = cp.x * g2.a + cp.y * g2.b + g2.c

    g1_mask = g1_values * cl_g1_side > 0
    g2_mask = g2_values * cl_g2_side > 0

    if image is not None:
        t = datetime.now()
        image_utils.save(f"{t:%Y-%m-%d_%H-%M-%S}_g1_mask.png", combine_heatmap(image, g1_mask))
        image_utils.save(f"{t:%Y-%m-%d_%H-%M-%S}_g2_mask.png", combine_heatmap(image, g2_mask))

    return np.logical_and(g2_mask, g1_mask)


def _project_cylinder_onto_corridor_landscape(
    centerline: geo.Line3D,
    radius: float,
    index_from_world: geo.CameraProjection,
    corridor_landscape: np.ndarray,
    hip_mask: Optional[np.ndarray],
    image: Optional[np.ndarray] = None,
) -> Tuple[float, Optional[np.ndarray]]:
    """Project onto the hip mask and integrate over the corridor landscape."""
    mask = _project_cylinder_region(centerline, radius, index_from_world, corridor_landscape.shape)

    # SO, actually, this hack to use the hip_mask to determine the boundaries of the corridor isn't
    # a great idea, because it removes the safeguard along the shaft.

    if hip_mask is not None:
        mask = mask & (hip_mask > 0.5)

    if image is not None:
        cl = index_from_world @ centerline
        t = datetime.now()
        # image_utils.save(f"{t:%Y-%m-%d_%H-%M-%S}_mask.png", combine_heatmap(image, mask))
        projection_image = combine_heatmap(
            combine_heatmap(image, corridor_landscape), mask, channel=2
        )
        corridor_image = image_utils.draw_line(image, cl, color=(255, 0, 0), thickness=1)
    else:
        projection_image = None
        corridor_image = None

    return float(np.sum(corridor_landscape[mask])), projection_image, corridor_image


t0 = datetime.now()

# TODO: possibly implement fully in numpy to allow for gradient to pass through.
def _adjust_corridor_obj(
    x: np.ndarray,
    corridor_landscape_1: np.ndarray,
    corridor_landscape_2: np.ndarray,
    hip_mask_1: np.ndarray,
    hip_mask_2: np.ndarray,
    index_from_world_1: geo.CameraProjection,
    index_from_world_2: geo.CameraProjection,
    image_1: Optional[np.ndarray] = None,
    image_2: Optional[np.ndarray] = None,
    radius: float = 3,
) -> float:
    """Objective function.

    The objective function gives the (negated) integral of the zero-centered landscape (the heatmap
    shifted such that the threshold value is at 0) over the projected area of the corridor cylinder
    in both images. One key thing is that we want the optimization to include the corridor up to the
    medial boundary of the hip, so

    Args:
        x: [p, q, r, s, t, u, radius] parameters of the 3D line, followed by the radius.
        corridor_landscape_1: The heatmap of the corridor in the first image.
        corridor_landscape_2: The heatmap of the corridor in the second image.
        index_from_world_1: The camera projection of the first image.
        index_from_world_2: The camera projection of the second image.
        startpoint_in_world: The approximate startpoint of the corridor in world coordinates. This is used
            to reduce search time for the startpoint of a gven with respect to both hip masks.
        direction_in_world: The approximate direction of the corridor in world coordinates. This is used to
            ensure the integral is taken in the right direction.

    Returns:
        The objective function value.

    """
    # log.debug(f"x: {x}")
    px: float = x[0]
    py: float = x[1]
    pz: float = x[2]
    theta: float = x[3]
    phi: float = x[4]

    p = geo.p(px, py, pz)
    d = geo.v(math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta))
    l = geo.line(p, d)

    val1, projection_image_1, corridor_image_1 = _project_cylinder_onto_corridor_landscape(
        l, radius, index_from_world_1, corridor_landscape_1, hip_mask_1, image_1
    )
    val2, projection_image_2, corridor_image_2 = _project_cylinder_onto_corridor_landscape(
        l, radius, index_from_world_2, corridor_landscape_2, hip_mask_2, image_2
    )
    out = -(val1 + val2)

    if projection_image_1 is not None and projection_image_2 is not None:
        t = datetime.now()
        projection_image = np.concatenate((projection_image_1, projection_image_2), axis=1)
        corridor_image = np.concatenate((corridor_image_1, corridor_image_2), axis=1)
        path = Path(f"adjustment/{t:%Y-%m-%d_%H-%M-%S_%f}_cylinder.png")
        image_utils.save(path, projection_image)
        # image_utils.save(f"adjustment/{t:%Y-%m-%d_%H-%M-%S_%f}_corridor.png", corridor_image)
        t0 = datetime.now()

    # log.debug(f"opt: {out}")
    return out


def downsize(
    corridor_heatmap: np.ndarray,
    hip_mask: Optional[np.ndarray],
    index_from_world: geo.CameraProjection,
    image_size: Tuple[int, int],
    image: Optional[np.ndarray] = None,
) -> Tuple[geo.FrameTransform, np.ndarray, np.ndarray, geo.CameraProjection, Optional[np.ndarray]]:

    # downsize the image and correct all .
    input_size = corridor_heatmap.shape
    index_from_input = geo.FrameTransform(
        [
            [image_size[1] / input_size[1], 0, 0],
            [0, image_size[0] / input_size[0], 0],
            [0, 0, 1],
        ]
    )
    index_from_world = geo.CameraProjection(
        index_from_input @ index_from_world.intrinsic.data,
        index_from_world.extrinsic,
    )
    corridor_heatmap = cv2.resize(corridor_heatmap, image_size, interpolation=cv2.INTER_AREA)
    if hip_mask is not None:
        hip_mask = cv2.resize(hip_mask, image_size, interpolation=cv2.INTER_AREA)
    if image is not None:
        image = cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)

    return index_from_input, corridor_heatmap, hip_mask, index_from_world, image


def adjust_corridor(
    corridor_heatmap_1: np.ndarray,
    corridor_heatmap_2: np.ndarray,
    hip_mask_1: Optional[np.ndarray],
    hip_mask_2: Optional[np.ndarray],
    index_from_world_1: geo.CameraProjection,
    index_from_world_2: geo.CameraProjection,
    startpoint_in_world: geo.Point3D,
    direction_in_world: geo.Vector3D,
    image_1: Optional[np.ndarray] = None,
    image_2: Optional[np.ndarray] = None,
    image_size: Tuple[int, int] = (384, 384),
    radius: float = 9,
    t_bound: float = 5,
    angle_bound: float = math.radians(5),
    threshold: float = 0.1,
) -> Tuple[geo.Point3D, geo.Vector3D, float]:
    """Adjust the triangulated corridor based on the heatmaps, optimizing for length.

    Note that this function assumes the index_from_world transform are properly aligned with the given images,
    so there may have to be some additional alignment step before calling this function, when using the LoopX
    class.

    Join projected images into an mp4 with:
        ffmpeg -framerate 5 -pattern_type glob -i '*.png' -c:v libx264 -pix_fmt yuv420p ../adjustment.mp4


    Args:
        corridor_heatmap_1: The heatmap of the corridor in the first image.
        corridor_heatmap_2: The heatmap of the corridor in the second image.
        index_from_world_1: The index from world coordinates to the first image.
        t_bound: The maximum translation of the corridor in world coordinates.
        angle_bound: The maximum rotation of the corridor in world coordinates (in radians).

    Returns:
        The startpoint, direction, and radius of the regressed corridor.
    """
    index_from_input_1, corridor_heatmap_1, hip_mask_1, index_from_world_1, image_1 = downsize(
        corridor_heatmap_1, hip_mask_1, index_from_world_1, image_size, image_1
    )
    index_from_input_2, corridor_heatmap_2, hip_mask_2, index_from_world_2, image_2 = downsize(
        corridor_heatmap_2, hip_mask_2, index_from_world_2, image_size, image_2
    )

    # Using https://www.wikiwand.com/en/Spherical_coordinate_system#/Cartesian_coordinates
    p0 = startpoint_in_world
    d0 = direction_in_world
    theta0 = np.arccos(direction_in_world[2])
    phi0 = np.arctan2(direction_in_world[1], direction_in_world[0])
    x0 = np.array([p0.x, p0.y, p0.z, theta0, phi0]).astype(np.float64)
    bounds = [
        (p0.x - t_bound, p0.x + t_bound),
        (p0.y - t_bound, p0.y + t_bound),
        (p0.z - t_bound, p0.z + t_bound),
        (theta0 - angle_bound, theta0 + angle_bound),
        (phi0 - angle_bound, phi0 + angle_bound),
    ]
    # log.debug(f"bounds: {bounds}")

    corridor_landscape_1 = corridor_heatmap_1 - threshold
    corridor_landscape_2 = corridor_heatmap_2 - threshold

    # os.environ["OMP_NUM_THREADS"] = "1"
    res = differential_evolution(
        _adjust_corridor_obj,
        x0=x0,
        args=(
            corridor_landscape_1,
            corridor_landscape_2,
            hip_mask_1,
            hip_mask_2,
            index_from_world_1,
            index_from_world_2,
            None,  # image_1,
            None,  # image_2,
            radius,
        ),
        bounds=bounds,
        tol=1,
        workers=1,
        polish=False,  # True causes segfaults.
    )

    # TODO: might need to find startpoint that's actually on teh bone.
    log.info(f"res: {res}")

    if not res.success:
        return startpoint_in_world, direction_in_world, radius

    log.info(f"Polishing...")
    # Refine, because the polish up above caused
    res = minimize(
        _adjust_corridor_obj,
        x0=res.x,
        args=(
            corridor_landscape_1,
            corridor_landscape_2,
            hip_mask_1,
            hip_mask_2,
            index_from_world_1,
            index_from_world_2,
            None,
            None,
            radius,
        ),
        method="Nelder-Mead",
        tol=0.1,
    )
    log.info(f"res: {res}")

    if not res.success:
        return startpoint_in_world, direction_in_world, radius

    # For making the adjustment figure.
    _adjust_corridor_obj(
        res.x,
        corridor_landscape_1,
        corridor_landscape_2,
        hip_mask_1,
        hip_mask_2,
        index_from_world_1,
        index_from_world_2,
        image_1,
        image_2,
        radius,
    )

    p = geo.p(res.x[0], res.x[1], res.x[2])
    theta = res.x[3]
    phi = res.x[4]
    r = radius
    d = geo.v(np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta))

    return p, d, r
