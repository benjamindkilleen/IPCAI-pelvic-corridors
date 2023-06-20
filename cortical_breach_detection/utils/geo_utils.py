from typing import Optional
from typing import Tuple
from typing import Union, List

import logging
import numpy as np
from deepdrr import geo
from deepdrr.utils import listify

log = logging.getLogger(__name__)


def project_on_segment(c: geo.Point, p: geo.Point, q: geo.Point) -> geo.Point:
    """Project the point `c` onto the line segment `pq`.

    Args:
        c (geo.Point): Point to project.
        p (geo.Point): One endpoint of the line segment.
        q (geo.Point): Other endpoint of the line segment.

    Returns:
        geo.Point: Projected point.

    """
    # Ensure types.
    c = geo.point(c)
    p = geo.point(p)
    q = geo.point(q)

    # Get the vector from p to q.
    v = q - p

    # Get the vector from p to c.
    u = p - c

    # Project u onto v.
    t = -v.dot(u) / v.dot(v)

    # Get the projected point.
    if t <= 0:
        return p
    elif t >= 1:
        return q
    else:
        return v * t + p


def fit_line(points: np.ndarray) -> Tuple[geo.Point, geo.Vector]:
    """Fit a line to points in 3D

    Args:
        points (np.ndarray): [N, 3] array of points.

    Returns:
        geo.Point3D, geo.Vector3D: Point, vector parameterization of the best fit line.
    """
    points = np.array(points)
    c = points.mean(axis=0)

    # Do an SVD on the mean-centered data.
    uu, dd, vv = np.linalg.svd(points - c[np.newaxis, :])

    # Now vv[0] contains the first principal component, i.e. the direction
    # vector of the 'best fit' line in the least squares sense.

    direction = geo.vector(vv[0]).hat()
    return geo.point(c), direction


def camera_point_from_index(
    pred_startpoint_in_index: geo.Point2D,
    d: float,
    camera3d_from_world: geo.FrameTransform,
    index_from_world: geo.Transform,
) -> geo.Point3D:
    # Get the vector (in camera space), going from the source to the landmark pixel,
    a_ray_hat = (camera3d_from_world @ index_from_world.inv @ pred_startpoint_in_index).hat()

    a_z = d
    a_y = a_ray_hat[1] * d / a_ray_hat[2]
    num = a_z * a_z + a_y * a_y
    den = a_ray_hat[2] * a_ray_hat[2] + a_ray_hat[1] * a_ray_hat[1]
    a_x = a_ray_hat[0] * np.sqrt(num / den)
    pred_startpoint_in_camera3d = geo.point(a_x, a_y, a_z)
    return pred_startpoint_in_camera3d


def distance_to_line(points: np.ndarray, c: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Get the distance to the line for a bunch of points.

    Args:
        points (np.ndarray): The points as an [N, D] array.
        c (np.ndarray): A point on the line.
        v (np.ndarray): A (unit?) vector pointing in the direction of the line.

    Returns:
        np.ndarray: The distance to the line for each point.
    """
    v = v / np.linalg.norm(v)
    diff = points - c
    return np.linalg.norm(diff - (diff @ v[:, np.newaxis]) * v[np.newaxis, :], axis=1)


def points_on_line(points: np.ndarray, c: np.ndarray, v: np.ndarray):
    v = v / np.linalg.norm(v)
    diff = points - c
    return c + (diff @ v[:, np.newaxis]) * v[np.newaxis, :]


def us_on_line(points: np.ndarray, c: np.ndarray, v: np.ndarray):
    v = v / np.linalg.norm(v)
    diff = points - c
    return (diff @ v[:, np.newaxis])[:, 0]


def geo_points_on_line(point: Union[geo.Point, List[geo.Point]], c: geo.Point, v: geo.Vector):
    """Deepdrr geo version, more flexible."""
    points = listify(point)
    ps = points_on_line(np.array(points), np.array(c), np.array(v))
    if isinstance(point, (list, tuple)):
        return [geo.point(p) for p in ps]
    elif isinstance(point, geo.Point):
        return geo.point(ps[0])
    else:
        raise TypeError(f"unexpected type: {type(point)}")


def geo_distance_to_line(point: Union[geo.Point, List[geo.Point]], c: geo.Point, v: geo.Vector):
    """Deepdrr geo version, more flexible."""
    points = listify(point)
    ps = distance_to_line(np.array(points), np.array(c), np.array(v))
    if isinstance(point, (list, tuple)):
        return [geo.point(p) for p in ps]
    elif isinstance(point, geo.Point):
        return ps[0]
    else:
        raise TypeError(f"unexpected type: {type(point)}")


def radius_of_circumscribing_cylinder(p: geo.Point3D, q: geo.Point3D, l: geo.Line3D) -> float:
    """Get the radius of a cylinder centered on (p -> q) that circumscribes the line.

    This is the radius of the cylinder with one end at p, the other at q, and large enough that the
    line enters one end and exits the other.

    Args:
        p (geo.Point3D): One end of the line segment.
        q (geo.Point3D): Other end of the line segment.
        l (geo.Line3D): The line.

    Returns:
        float: Radius of the circumscribing cylinder.
    """

    # Normal of ends.
    n = q - p

    pl_p = geo.plane(p, n)
    pl_q = geo.plane(q, n)

    # Points on each cylinder end through which the line passes.
    lp = pl_p.meet(l)
    lq = pl_q.meet(l)

    # Radius of the cylinder is the max of the two distances.
    return max((lp - p).norm(), (lq - q).norm())
