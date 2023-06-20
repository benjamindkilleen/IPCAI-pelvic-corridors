from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from deepdrr import geo
import cortical_breach_detection.viewpoint_planning as vp
from cortical_breach_detection.utils import combine_heatmap, nn_utils


def line_heatmap(l: np.ndarray, scale: float, size: Tuple[int, int]) -> np.ndarray:
    """Create a heatmap of a line. Does not normalize.

    Args:
        l (np.ndarray): a line in homogeneous coordinates, which expects homogeneous points in (column, row) form.
        scale (float): the scale of the heatmap.
        size (Tuple[int]): the size of the heatmap.

    Returns:
        np.ndarray: a heatmap of the line.

    """
    # Convert line to point, point in row, column form.

    # Direction of line
    v = vp.get_direction_of_line(l)
    p1 = vp.meet(l, v)
    p2 = p1 + v
    c1, r1 = vp.real_from_homogeneous(p1)
    c2, r2 = vp.real_from_homogeneous(p2)
    return nn_utils.line_heatmap_numpy(r1, c1, r2, c2, scale, size)


def fun(l: np.ndarray, h_pred: np.ndarray, scale: float, size: Tuple[int, int]) -> float:
    """Objective function for -log(p(l|h))

    Args:
        l (np.ndarray): [a,b,c] line in homogeneous coordinates.
        h_pred (np.ndarray): Predicted heatmap.
        scale (float): Scale of heatmap.
        size (Tuple[int, int]): Size of heatmap.

    Returns:
        float: Objective function value.
    """
    h = line_heatmap(l, scale, size)
    C = np.sum(h)
    return np.sum(np.log(h_pred) - np.log(h / C))


def jac(l: np.ndarray, h_pred: np.ndarray, scale: float, size: Tuple[int, int]) -> np.ndarray:
    """Jacobian of objective function.

    Args:
        l (np.ndarray): [a,b,c] line in homogeneous coordinates.
        h_pred (np.ndarray): Predicted heatmap.
        scale (float): Scale of heatmap.
        size (Tuple[int, int]): Size of heatmap.

    Returns:
        np.ndarray: Jacobian of objective function.
    """
    return None


def fit_single_line(heatmap: np.ndarray):

    size = heatmap.shape
    scale = size[0] / 80

    heatmap_max = np.max(heatmap)
    heatmap_min = np.min(heatmap)
    h_pred = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)

    # minimize(fun, np.array([0, 0, 0]), args=(h_pred, scale, size), jac=jac, method='L-BFGS-B')
    res = minimize(fun, np.array([0, 0, 0]), args=(h_pred, scale, size))
    l = res.x

    plt.imshow(h_pred)
    r = vp.get_direction_of_line(l)
    p1 = vp.meet(l, r)
    p2 = p1 + 200 * r
    x1, y1 = vp.real_from_homogeneous(p1)
    x2, y2 = vp.real_from_homogeneous(p2)
    plt.plot([x1, x2], [y1, y2], "r-")
    plt.savefig("images/heatmap_test.png")


def main():
    
    pass


if __name__ == "__main__":
    main()
