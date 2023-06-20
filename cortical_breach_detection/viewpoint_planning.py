import logging
import math
from typing import Tuple


import numpy as np

# import autograd.numpy as np


log = logging.getLogger(__name__)


# Arrays with shape (3,) are points or lines in P^2
# Arrays with shape (3,) are points or planes in P^3
# Arrays with shape (6,) are lines in P^3


def real_from_homogeneous(x: np.ndarray) -> np.ndarray:
    """Get the real part of a homogeneous vector.

    Args:
        x (np.ndarray): Homogeneous vector.

    Returns:
        np.ndarray: Real part of the homogeneous vector.
    """
    assert x[-1] != 0
    return x[:-1] / x[-1]


def skew(v: np.ndarray) -> np.ndarray:
    """Get the skew-symmetric matrix of a vector.

    Args:
        v (np.ndarray): Shape (3,) vector in P^3.

    Returns:
        np.ndarray: Skew-symmetric matrix of the vector.
    """
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def primal_from_line(L: np.ndarray) -> np.ndarray:
    """Get the primal matrix of the line.

    Args:
        L (np.ndarray): Shape (6,) line in P^3.

    Returns:
        np.ndarray: Primal matrix of the line.
    """
    p, q, r, s, t, u = L

    return np.array(
        [
            [0, p, -q, r],
            [-p, 0, s, -t],
            [q, -s, 0, u],
            [-r, t, -u, 0],
        ]
    )


def line_from_primal(LP: np.ndarray) -> np.ndarray:
    """Compute the line from its primal matrix.

    Args:
        LP (np.ndarray): Primal matrix of the line.

    Returns:
        np.ndarray: Line in P^3.
    """
    assert LP.shape == (4, 4)
    return np.array([LP[0, 1], -LP[0, 2], LP[0, 3], LP[1, 2], -LP[1, 3], LP[2, 3]])


def dual_from_line(L: np.ndarray) -> np.ndarray:
    """Get the dual matrix of the line.

    Args:
        L (np.ndarray): Shape (6,) line in P^3.

    Returns:
        np.ndarray: Dual matrix of the line.
    """
    p, q, r, s, t, u = L

    return np.array(
        [
            [0, -u, -t, -s],
            [u, 0, -r, -q],
            [t, r, 0, -p],
            [s, q, p, 0],
        ]
    )


def line_from_dual(LK: np.ndarray) -> np.ndarray:
    """Compute the line from its dual matrix.

    Args:
        LK (np.ndarray): Dual matrix of the line.

    Returns:
        np.ndarray: Line in P^3.
    """
    return np.array(
        [
            LK[3, 2],
            LK[3, 1],
            LK[2, 1],
            LK[3, 0],
            LK[2, 0],
            LK[1, 0],
        ]
    )


def check_line(L: np.ndarray) -> None:
    """Check that the line is valid.

    Based on (*) in https://dl.acm.org/doi/pdf/10.1145/965141.563900.

    Args:
        L (np.ndarray): Shape (6,) line in P^3.

    Returns:
        None.
    """
    assert L.shape == (6,)
    p, q, r, s, t, u = L
    star = p * u - q * t + s * r
    if not np.isclose(star, 0):
        log.warning(f"(*) condtion failed: {star}")
    # assert np.isclose(star, 0), f"(*) condition not satisfied: {star}"


def meet(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape == (3,) and b.shape == (3,):
        # Intersection of two lines in P^2.
        c = np.cross(a, b)
        assert c[2] != 0
        return c
    elif a.shape == (4,) and b.shape == (4,):
        # Intersection of two planes in P^3.
        a1, b1, c1, d1 = a
        a2, b2, c2, d2 = b
        L = np.array(
            [
                -(a1 * b2 - a2 * b1),  # p
                a1 * c2 - a2 * c1,  # q
                -(a1 * d2 - a2 * d1),  # r
                -(b1 * c2 - b2 * c1),  # s
                b1 * d2 - b2 * d1,  # t
                -(c1 * d2 - c2 * d1),  # u
            ]
        )
        check_line(L)
        return L
    elif a.shape == (6,) and b.shape == (4,):
        # Intersection of a line and a plane in P^3.
        return dual_from_line(a) @ b
    else:
        raise ValueError(f"Invalid shapes: {a.shape} and {b.shape}.")


def join(a: np.ndarray, b: np.ndarray, check: bool = True) -> np.ndarray:
    if a.shape == (3,) and b.shape == (3,):
        # Line joining two points in P^2.
        l = np.cross(a, b)
        return l
    elif a.shape == (4,) and b.shape == (4,):
        # Line joining two points in P^3.
        ax, ay, az, aw = a
        bx, by, bz, bw = b
        L = np.array(
            [
                az * bw - aw * bz,  # p
                ay * bw - aw * by,  # q
                ay * bz - az * by,  # r
                ax * bw - aw * bx,  # s
                ax * bz - az * bx,  # t
                ax * by - ay * bx,  # u
            ]
        )
        if check:
            check_line(L)
        return L
    elif a.shape == (6,) and b.shape == (4,):
        # Plane through a line and a point in P^3.
        return b.T @ primal_from_line(a)
    else:
        raise ValueError(f"Invalid shapes: {a.shape} and {b.shape}.")


def plane_from_point_normal(p: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Get the plane through a point and normal.

    Args:
        p (np.ndarray): Shape (4,) point in P^2.
        n (np.ndarray): Shape (4,) normal vector.

    Returns:
        np.ndarray: Shape (4,) plane in P^3.
    """
    assert p.shape == (4,)
    assert n.shape == (4,)
    return np.array([*n[:3], -np.dot(p, n)])


def normal_from_plane(p: np.ndarray) -> np.ndarray:
    """Get the normal vector of a plane.

    Args:
        p (np.ndarray): Shape (4,) plane in P^3.

    Returns:
        np.ndarray: Normal vector of the plane.
    """
    n = np.array([p[0], p[1], p[2], 0])
    return n / np.linalg.norm(n)


def angulation(a: np.ndarray, b: np.ndarray) -> float:
    """Get the angulation between two vectors.

    Args:
        a (np.ndarray): Shape (3,) vector in P^3.
        b (np.ndarray): Shape (3,) vector in P^3.

    Returns:
        float: Angle between a and b.
    """
    if np.linalg.norm(a) == 0:
        return 0
    if np.linalg.norm(b) == 0:
        return 0
    return np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def angle_between_planes(p1: np.ndarray, p2: np.ndarray) -> float:
    """Get the angle between two planes.

    Args:
        p1 (np.ndarray): Shape (4,) plane in P^3.
        p2 (np.ndarray): Shape (4,) plane in P^3.

    Returns:
        float: Angle between the two planes.
    """
    return angulation(normal_from_plane(p1), normal_from_plane(p2))


def rodriguez_rotation(k: np.ndarray, theta: float) -> np.ndarray:
    """Get the rotation matrix of a rotation around an axis.

    Args:
        k (np.ndarray): Shape (3,) rotation axis R^3.
        theta (float): Rotation angle.

    Returns:
        np.ndarray: (3,3) Rotation matrix.
    """
    k_hat = k / np.linalg.norm(k)
    K = skew(k_hat)
    return np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * K @ K


def homogeneous_rodriguez_rotation(k: np.ndarray, theta: float) -> np.ndarray:
    """Get the homogeneous rotation matrix of a rotation around an axis.

    Args:
        k (np.ndarray): Shape (4,) rotation axis P^3.
        theta (float): Rotation angle.

    Returns:
        np.ndarray: (4,4) Homogeneous transform matrix containing the rotation.
    """
    assert np.isclose(k[-1], 0)
    R = rodriguez_rotation(k[:3], theta)
    return np.array(
        [
            [R[0, 0], R[0, 1], R[0, 2], 0],
            [R[1, 0], R[1, 1], R[1, 2], 0],
            [R[2, 0], R[2, 1], R[2, 2], 0],
            [0, 0, 0, 1],
        ]
    )


def angles_from_principle_ray(p: np.ndarray) -> Tuple[float, float]:
    """Get the angles of a principle ray.

    NOTE: this follows the convention of DeepDRR/Kausch. Preuhs et al. defines

    alpha = atan2(n_x, -n_y)
    beta = asin(n_z)

    Args:
        p (np.ndarray): Shape (3,) principle ray in P^3.

    Returns:
        np.ndarray: (3,) Angles of the principle ray.
    """
    n = p / np.linalg.norm(p)
    alpha = math.asin(n[2])
    beta = -math.atan2(n[0], -n[1])
    return alpha, beta


def angles_from_projection(P: np.ndarray) -> Tuple[float, float]:
    """Get the angles of a projection.

    Args:
        P (np.ndarray): Shape (3,) projection in P^3.

    Returns:
        np.ndarray: (3,) Angles of the projection.
    """
    return angles_from_principle_ray(principle_ray_from_projection(P))


def get_direction_of_line(l: np.ndarray) -> np.ndarray:
    """Get the direction of a line.

    Args:
        l (np.ndarray): Shape (6,) line in P^3.

    Returns:
        np.ndarray: (4,) Direction of the line.
    """
    if l.shape == (6,):
        p, q, r, s, t, u = l
        r = np.array([s, q, p, 0])
        return r / np.linalg.norm(r)
    elif l.shape == (3,):
        a, b, c = l
        v = np.array([b, -a, 0])
        v = v / np.linalg.norm(v)
        return v
    else:
        raise ValueError(f"Invalid shape: {l.shape}.")


def get_point_on_line(l: np.ndarray) -> np.ndarray:
    """Get an arbitrary point on a line.

    Do this by finding the intersection of the plane with the same normal as the line.

    Args:
        l (np.ndarray): Shape (6,) line in P^3.
        t (float): Parameter of the point on the line.

    Returns:
        np.ndarray: (3,) Point on the line.
    """
    assert l.shape == (6,) or l.shape == (3,)
    # Direction can also be interpreted as a plane in the same direction.
    r = get_direction_of_line(l)
    return meet(l, r)


def principle_ray_from_projection(P: np.ndarray) -> np.ndarray:
    """Get the principle ray from a projection matrix.

    Args:
        P (np.ndarray): (3,4) Projection matrix.

    Returns:
        np.ndarray: (3,) Principle ray direction in R^3.
    """
    assert P.shape == (3, 4)
    m = P[2, :3] / np.linalg.norm(P[2, :3])
    return np.array([m[0], m[1], m[2], 0])


def plan_viewpoint(
    c1: np.ndarray, c2: np.ndarray, P1: np.ndarray, xi: float = math.pi / 6
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a second viewpoint to view the line between c1 and c2.

    NOTE: the "world" frame here needs to be the isocentric frame of the C-arm, aligned with the
    axes pointing as indicated in Fig 1 of the paper.

    Based on https://doi.org/10.1007/s11548-018-1763-1.

    Args:
        c1 (np.ndarray): (3,) Homogeneous coordinates of the first point in P^2.
        c2 (np.ndarray): (3,) Homogeneous coordinates of the second point in P^2.
        P1 (np.ndarray): (3, 4) Projection matrix of the first viewpoint.
        xi (float): How much to rotate by in radians. Defaults to pi/6.

    Returns:
        np.ndarray: (3,) The principle ray direction of the second viewpoint in R^3.
        np.ndarray: (4,) The translation vector to perform to the device, in P^3.
    """
    assert c1.shape == (3,)
    assert c2.shape == (3,)
    assert P1.shape == (3, 4)

    c1 = c1.astype(np.float64)
    c2 = c2.astype(np.float64)
    P1 = P1.astype(np.float64)

    # Line in the image
    l = join(c1, c2)

    # X-ray source (camera center) in world.
    s = meet(meet(P1[0], P1[1]), P1[2])
    P1_inv = np.linalg.pinv(P1)

    # Get the plane containing all points in P^3 that are projected onto the line.
    e_R = P1.T @ l
    # e_R = join(join(P1_inv @ c1, P1_inv @ c2), s) # mathematically the same as above

    # Backprojection of the line, i.e. the 3D line in the image plane.
    LK = P1_inv @ skew(l) @ P1_inv.T
    L = line_from_dual(LK)
    # pi_inf = np.array([0, 0, 0, 1])
    # r = meet(L, pi_inf)
    r = get_direction_of_line(L)

    # Get the normal of the plane containing the line in the image but orthogonal to the principle
    # ray. Essentially, this is the plane containing points that would fall in the line in an
    # orthogonal projection from the same direction as P1.
    # Here, we construct it by the join of L with a point offset from L by the principle ray.
    m = principle_ray_from_projection(P1)
    e_L = join(L, get_point_on_line(L) + m)
    phi = angle_between_planes(e_L, e_R)
    log.debug(f"TODO: determine if phi should be negated? phi: {phi}")
    # TODO: determine if phi should be negated?

    # log.debug(f"phi: {np.degrees(phi)}")

    ######## Isocenter rotation ########

    # The alignment transform.
    P1_align = P1 @ homogeneous_rodriguez_rotation(r, phi)

    ######## Isocenter offset correction ########

    # Source position of the alignment rotation
    s_a = meet(meet(P1_align[0], P1_align[1]), P1_align[2])

    # Distance between e_L and the source position
    # TODO: determine if this is correct, or if the version in the paper is correct (despite typos)
    # This seems to check out more fundamentally.
    d = (e_L.T @ s_a) / (np.linalg.norm(e_L[:3]) * s_a[3])
    # TODO: determine if this is the right direction

    log.debug("TODO: determine if d should be negated")
    # Negation depends on orientation sp_a - sp.

    # Translation
    t = -d * normal_from_plane(e_L)
    # log.debug(f"translation: {t}")

    P2 = P1_align @ homogeneous_rodriguez_rotation(r, xi)

    ray = principle_ray_from_projection(P2)
    return ray, t
    # alpha, beta = angles_from_projection(P2)
    # return alpha, beta, t
