import itertools
import logging
import math
from pathlib import Path
from time import time
from typing import Optional
from typing import Tuple

import matplotlib as mpl
from deepdrr import geo
from deepdrr import MobileCArm
from deepdrr.utils import tuplify

mpl.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from PIL import Image
from rich.logging import RichHandler
from scipy import ndimage
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
from scipy.special import binom
from scipy.stats import special_ortho_group
from skimage import color
from skimage import data
from skimage import img_as_ubyte
from skimage.draw import ellipse_perimeter
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.transform import rescale
from deepdrr.annotations import LineAnnotation

from cortical_breach_detection import utils
from cortical_breach_detection.deepdrr_tools.wire_guide import WireGuide

log = logging.getLogger("cortical_breach_detection")
logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
log.setLevel(logging.DEBUG)


def ellipse(
    theta: np.ndarray, r: float, c: float, r_radius: float, c_radius: float, alpha: float = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Get the points on the ellipse parameterized by the angle theta from the x-axis.

    Follows the conventions of skimage.draw.

    Args:
        theta (np.ndarray): length N array of thetas.
        yc (float): second axis center.
        xc (float): first axis center.
        a (float): first axis radius.
        b (float): second axis radius.
        alpha (float): Rotation of the ellipse from the first axis in radians.

    Returns:
        np.ndarray: [N, 2] array of (x, y) points on the ellipse.
    """
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_alpha = np.sin(alpha)
    cos_alpha = np.cos(alpha)
    rs = r_radius * cos_theta * cos_alpha - c_radius * sin_theta * sin_alpha + r
    cs = r_radius * cos_theta * sin_alpha + c_radius * sin_theta * cos_alpha + c
    return rs, cs


# def find_disks(image: np.ndarray, num_expected: int = 2, value: float = 0.5, pad: int = 3) -> np.ndarray:
#     assert image.dtype == np.uint8
#     labeled, num_features = ndimage.label(image)
#     if num_features != num_expected:
#         raise NotImplementedError

#     object_slices = ndimage.find_objects(labeled)

#     fig, ax = plt.subplots(
#         ncols=4,
#         nrows=num_features,
#     )
#     for i in range(num_features):
#         l = i + 1
#         si, sj = object_slices[i]
#         si = slice(max(0, si.start - pad), min(image.shape[0], si.stop + pad))
#         sj = slice(max(0, sj.start - pad), min(image.shape[1], sj.stop + pad))
#         ci = (si.start + si.stop) / 2 - si.start
#         cj = (sj.start + sj.stop) / 2 - sj.start
#         patch = image[si, sj]
#         ax[i, 0].imshow(patch, interpolation="none")
#         ax[i, 0].plot(cj, ci, "rx")

#         flipped = patch.max() - patch
#         f_labeled, f_num_features = ndimage.label(flipped)
#         inner_label = f_labeled[f_labeled.shape[0] // 2, f_labeled.shape[1] // 2]

#         inner_ellipse = (f_labeled == inner_label).astype(np.uint8)
#         outer_ellipse = np.where(inner_ellipse > 0, 1, patch / 255).astype(np.uint8)
#         ax[i, 1].imshow(inner_ellipse, interpolation="none")
#         ax[i, 2].imshow(outer_ellipse, interpolation="none")

#         (x, y, x_radius, y_radius, rotation), _ = find_ellipse(outer_ellipse)

#         ax[i, 3].imshow(patch, interpolation="none")
#         theta = np.arange(0, 2 * np.pi, np.radians(5))
#         rs, cs = ellipse(theta, x, y, x_radius, y_radius, rotation)
#         ax[i, 3].plot(rs, cs, "r-")

#         # inner ellipse
#         (x, y, x_radius, y_radius, rotation), _ = find_ellipse(inner_ellipse)
#         rs, cs = ellipse(theta, x, y, x_radius, y_radius, rotation)
#         ax[i, 3].plot(rs, cs, "b-")

#     plt.savefig("images/ellipse_detection.png", dpi=200)


def find_ellipses(image: np.ndarray, value: float = 0.5) -> np.ndarray:
    # Maybe use this: https://www.hindawi.com/journals/tswj/2014/481312/
    # https://github.com/AlanLuSun/High-quality-ellipse-detection
    # Python wrappers: https://github.com/Li-Zhaoxi/AAMED

    # Write it myself:
    # * Extract approximate region of image with high values. Should be fairly localized.
    # * Initialize with rough Hough estimate from skimage or opencv on the edges, then for each estimated ellipse
    # * Take |image - value| and do gradient descent on the sum of interpolated points along the ellipse, using pytorch? Could work.

    # Returns an [N, 4] array with ellipse parameters [x_center, y_center, ]

    # Load picture, convert to grayscale and detect edges
    image_gray = image
    edges = canny(image_gray, sigma=3.0, low_threshold=0.7, high_threshold=0.8)
    plt.imshow(edges)
    plt.savefig("images/ellipse_edges.png")

    # Perform a Hough Transform
    # The accuracy corresponds to the bin size of a major axis.
    # The value is chosen in order to get a single high accumulator.
    # The threshold eliminates low accumulators
    result = hough_ellipse(edges, accuracy=2, threshold=4, min_size=5, max_size=120)
    result.sort(order="accumulator")

    # Estimated parameters for the ellipse
    # best = list(result[-1])
    edges = color.gray2rgb(img_as_ubyte(edges))
    image_rgb = color.gray2rgb(image)
    print(result)
    for r in result[-10:]:
        best = list(r)
        yc, xc, a, b = [int(round(x)) for x in best[1:5]]
        orientation = best[5]

        # Draw the ellipse on the original image
        cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
        image_rgb[cy, cx] = (0, 0, 255)
        # Draw the edge (white) and the resulting ellipse (red)
        edges[cy, cx] = (250, 0, 0)

    fig2, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(8, 4), sharex=True, sharey=True)
    ax1.set_title("Original picture")
    ax1.imshow(image_rgb)

    ax2.set_title("Edge (white) and result (red)")
    ax2.imshow(edges)

    plt.savefig("images/ellipse_detection.png")


def test_something():
    xs, ys = ellipse(np.arange(0, 2 * np.pi, np.radians(4)), 0, 0, 1, 1)
    plt.plot(xs, ys, "b.")
    plt.savefig(f"images/test_something.png")


THETAS = np.arange(0, 2 * np.pi, np.radians(3))


def main():

    # Fully perpendicular
    image_path = "/home/benjamin/datasets/CTPelvic1K/projections_06-views_12-trajs_-5-00-05-10-15_oop_WireGuide/dataset6_CLINIC_0001_left/001_obturator_oblique/p--05_t-00_000_breach_kwire-guide-mask.png"
    info_path = "/home/benjamin/datasets/CTPelvic1K/projections_06-views_12-trajs_-5-00-05-10-15_oop_WireGuide/dataset6_CLINIC_0001_left/001_obturator_oblique/p--05_t-00_000_breach_info.json"

    # More at an angle
    # image_path = "/home/benjamin/datasets/CTPelvic1K/projections_06-views_12-trajs_-5-00-05-10-15_oop_WireGuide/dataset6_CLINIC_0001_left/000_anteroposterior/p--05_t-00_000_breach_kwire-guide-mask.png"
    # info_path = "/home/benjamin/datasets/CTPelvic1K/projections_06-views_12-trajs_-5-00-05-10-15_oop_WireGuide/dataset6_CLINIC_0001_left/000_anteroposterior/p--05_t-00_000_breach_info.json"

    info = utils.load_json(info_path)
    annotation = LineAnnotation.from_markup(
        info["annotation_path"],
        anatomical_from_world=info["anatomical_from_world"],
        anatomical_coordinate_system=info["anatomical_coordinate_system"],
    )
    startpoint = geo.point(info["startpoint"])
    endpoint = geo.point(info["endpoint"])

    carm = MobileCArm(
        min_alpha=-40,
        max_alpha=110,
        min_beta=-90,
        max_beta=90,
        degrees=True,
        sensor_height=768,
        sensor_width=768,
        pixel_size=0.388,
        enforce_isocenter_bounds=False,
    )
    carm.move_to(**info["view"])
    left_side = "left" in info["annotation"]

    wire_guide = WireGuide()
    wire_guide.align(startpoint, endpoint, progress=info["kwire_guide_progress"])
    index_from_world = carm.get_camera_projection().index_from_world
    camera3d_from_world = carm.get_camera_projection().camera3d_from_world
    log.debug(f"gt tip camera: {camera3d_from_world @ wire_guide.tip_in_world}")
    vx, vy, vz = (camera3d_from_world @ (wire_guide.base_in_world - wire_guide.tip_in_world)).hat()
    theta = math.acos(vz)
    phi = math.atan2(vy, vx)
    log.debug(f"gt theta, phi: {theta, phi}, degrees: {np.degrees([theta, phi])}")
    # for c in wire_guide.centers_in_world:
    #     c_index = index_from_world @ c
    #     plt.plot(*c_index, 'g+')

    image = np.array(Image.open(image_path))
    washer_centers_in_index, washer_centers = reconstruct_wire_guide(
        image,
        num_expected=3,
        left_side=left_side,
        pixel_size=carm.pixel_size,
        source_to_detector_distance=carm.source_to_detector_distance,
        camera3d_from_world=geo.FrameTransform(info["camera3d_from_world"]),
        index_from_world=geo.Transform(info["index_from_worlds"], _inv=info["world_from_indexes"]),
        # washer_outer_radius=7.865,
        # washer_height=2.644,
        fast=True,
    )

    # plt.imshow(image, interpolation="none")
    # for disk in disks:
    #     if "inner" in disk:
    #         rs, cs = ellipse(THETAS, *disk["inner"])
    #         plt.plot(rs, cs, "b-", linewidth=0.5)

    #     if "outer" in disk:
    #         log.debug("got params: {}".format(disk["outer"]))
    #         rs, cs = ellipse(THETAS, *disk["outer"])
    #         plt.plot(rs, cs, "r-", linewidth=0.5)

    # plt.savefig("images/ellipse_detection_full_image.png", dpi=300)


if __name__ == "__main__":
    # test_something()
    main()
