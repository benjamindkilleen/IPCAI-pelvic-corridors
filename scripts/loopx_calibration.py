import csv
import math
from autograd import value_and_grad
from datetime import datetime
from pathlib import Path
from typing import List
import numpy as np
import scipy.optimize
from scipy.spatial.transform import Rotation as R
import click
from deepdrr.utils import data_utils, dicom_utils, image_utils
import cv2
import logging
from rich.logging import RichHandler
import matplotlib.pyplot as plt
from rich.progress import track
from loopx_objective import inverse, loopx_objective
import pydicom

from cortical_breach_detection.utils import onedrive_utils, plus_utils
from cortical_breach_detection.loopx import Acquisition


logging.basicConfig(
    level="DEBUG",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger("loopx_calibration")
logging.getLogger("matplotlib").setLevel(logging.WARNING)

# Always go from front to back. (M,3) array of sphere positions in the pointer frame.   `)
spheres_in_pointer = np.array(
    [[43.96, 9.77, 60.70], [-10.76, -1.04, -0.56], [-33.20, -8.73, -60.14]]
)

# Read directly off machine, for each image take multiple measurements and average the quats/positions
# Now, what's actually needed are just the device_time, tracker_from_loopx, and tracker_from_pointer transforms.
# That's it.
example_data = [
    dict(
        device_time="2022-10-08 23:37:06.091",  # time of image capture, as seen on machine
        source_angle=180,  # degrees
        detector_angle=0,  # degrees (for checking 180 deg separation)
        gantry_tilt=0,  # degrees
        tracker_from_loopx=[],  # list of strings containing 4x4 transformation matrices
        tracker_from_pointer=[],  # list of strings containing 4x4 transformation matrices
    ),
]


def detect_circles(image: np.ndarray) -> np.ndarray:
    """Detect circles in an image.

    Args:
        image: (H,W) image

    Returns:
        (N,3) array of (x,y) for each circle detected.
    """
    original_image = image.copy()
    image = image_utils.as_uint8(image)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.medianBlur(image, 5)
    H, W = image.shape
    circles = cv2.HoughCircles(
        image,
        cv2.HOUGH_GRADIENT,
        1,
        20,  # min distance between circles
        param1=100,
        param2=30,
        minRadius=40,
        maxRadius=100,
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(original_image, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv2.circle(original_image, center, radius, (255, 0, 255), 3)

    # image_utils.save("images/circles.png", original_image)
    return circles[0, :, :2]


BAD_IMAGES = {
    "31 20221008220000163 (2022-10-09 0031).png",
    "049737_20221009003116428.dcm",
    # "044497_20221009000151243",
    # "045568_20221009000752082",
}


def _load_image_data(data_path: Path, images_dir: Path, plus_dir: Path) -> List[dict]:
    """Load images and data from a list of calibration data.

    Args:
        data_path: Path to the csv containing device times and tracking filenames.
        images_dir: Path to the directory containing the images.
        plus_dir: Path to the directory containing the plus tracking data.

    Returns:
        list of calibration data with images and spheres_in_image added.
    """
    dicom_paths_ = sorted(list(images_dir.glob("**/*.dcm")))
    dicom_paths = []
    dicom_times = []
    for p in dicom_paths_:
        t = dicom_utils.get_time(p)
        if t is not None:
            dicom_paths.append(p)
            dicom_times.append(t)

    file = open(data_path, "r")
    reader = csv.DictReader(file)

    image_data = dict(
        image_size=[],
        points_in_image=[],
        image_path=[],
        pixel_size=[],
        index_from_isocenter=[],
        tracker_from_pointer=[],
        tracker_from_loopx=[],
    )

    for i, row in enumerate(track(reader, description="Processing images...")):
        # Check if the corresponding image exists in the path
        device_timestamp = row["device_time"]
        device_time = datetime.fromisoformat(device_timestamp)
        dts = [abs((device_time - t).total_seconds()) for t in dicom_times]
        idx = np.argmin(dts)
        if (device_time - dicom_times[idx]).total_seconds() > 2:
            log.warning(f"Could not find image for {device_timestamp}")
            continue
        dicom_path = dicom_paths[idx]

        if (
            dicom_path.name in BAD_IMAGES
            or dicom_path.stem in BAD_IMAGES
            or device_timestamp in BAD_IMAGES
        ):
            log.info(f"Skipping '{dicom_path.name}' because it is bad")
            continue

        log.info(f"Processing {dicom_path.name}")
        acquisition = Acquisition(dicom_path)
        image = acquisition.image
        image_size = image.shape[:2]
        points_in_image = detect_circles(image)
        assert points_in_image.shape[0] == 3

        # Sort the points in the image by their x coordinate, in descending order,
        # so that the points are front-to-back. (pointer was oriented same way in all images.)
        points_in_image = points_in_image[np.argsort(points_in_image[:, 0])[::-1]]

        tracking_filename = row["tracking_filename"]
        tracking_path = plus_dir / "data" / tracking_filename
        if not tracking_path.exists():
            log.warning(f"Could not find tracking file {tracking_path}")
            continue
        frames = plus_utils.average_transforms(tracking_path)
        tracker_from_pointer = frames[("tracker", "pointer")]
        tracker_from_loopx = frames[("tracker", "loop_x")]

        image_data["image_size"].append(image_size)
        image_data["points_in_image"].append(points_in_image)
        image_data["image_path"].append(dicom_path)
        image_data["pixel_size"].append(acquisition.pixel_size)
        image_data["index_from_isocenter"].append(acquisition.index_from_isocenter)
        image_data["tracker_from_pointer"].append(tracker_from_pointer)
        image_data["tracker_from_loopx"].append(tracker_from_loopx)

        image = image_utils.ensure_cdim(image_utils.as_uint8(image))

        cs = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
        ]
        for j, (x, y) in enumerate(points_in_image):
            cv2.circle(image, (x, y), 25, cs[j], 3)
        image_utils.save(f"images/{j:02d}_{dicom_path.stem}.png", image)

    image_data = {k: np.array(v) for k, v in image_data.items()}
    file.close()
    return image_data


def load_image_data(
    data_path: Path, images_dir: Path, plus_dir: Path, cache: bool = True
) -> List[dict]:
    """Load the image data. If it has not yet been collected, analyze images."""
    image_data_path = Path("data") / "image_data.npz"
    if image_data_path.exists() and cache:
        log.info("Loading image data from file")
        image_data = np.load(image_data_path, allow_pickle=True)
        image_data = dict(image_data)
    else:
        log.info("Analyzing images")
        image_data = _load_image_data(data_path, images_dir, plus_dir)
        np.savez(image_data_path, **image_data)

    return image_data


def fromstring(s: str) -> np.ndarray:
    """Convert a string representation of a 4x4 matrix into a numpy array."""
    return np.array(s.split()).astype(float).reshape(4, 4)


def tostring(A: np.ndarray) -> str:
    """Convert a 4x4 matrix into a string representation with newlines."""
    formatter = {"float_kind": lambda x: "%.8f" % x}
    lines = [
        np.array2string(A[i], separator=" ", formatter=formatter)[1:-1] for i in range(A.shape[0])
    ]
    return "\n".join(lines)


def fromstrings(ss: List[str]) -> np.ndarray:
    """Convert a list of strings into a (N, 4, 4)."""
    return np.array([fromstring(s) for s in ss])


def get_quat(F: np.ndarray) -> np.ndarray:
    """Get quaternion of the transformation."""
    q = R.from_matrix(F[:3, :3]).as_quat()
    return q


def get_rotvec(F: np.ndarray) -> np.ndarray:
    """Get rotation vector of the transformation."""
    return R.from_matrix(F[:3, :3]).as_rotvec()


def get_pos(F: np.ndarray) -> np.ndarray:
    """Get position of the transformation."""
    p = F[:3, 3]
    return p


def inverse(F: np.ndarray) -> np.ndarray:
    """Inverse of a 4x4 transformation matrix"""
    R = F[:3, :3]
    t = F[:3, 3]
    return np.vstack((np.hstack((R.T, -R.T @ t[:, np.newaxis])), np.array([0, 0, 0, 1])))


def average_transforms(Fs: np.ndarray) -> np.ndarray:
    """Average a list of 4x4 transformation matrices."""
    # Get the euler angles
    rotvecs = np.array([R.from_matrix(F[:3, :3]).as_rotvec() for F in Fs])

    # get the average euler angles
    rotvec = np.mean(rotvecs, axis=0)

    # Get the average position
    pos = np.mean([F[:3, 3] for F in Fs], axis=0)

    # Get the average rotation matrix
    rot = R.from_rotvec(rotvec).as_matrix()

    # Get the average transformation matrix
    avg_F = np.eye(4)
    avg_F[:3, :3] = rot
    avg_F[:3, 3] = pos

    return avg_F


def average_transform_fromstrings(ss: List[str]) -> np.ndarray:
    """Average a list of 4x4 transformation matrices represented as strings."""
    Fs = fromstrings(ss)
    return average_transforms(Fs)


def calibrate_loopx(
    focal_length: float,
    xs: np.ndarray,
    centers: np.ndarray,
    theta_S: np.ndarray,
    F_TL: np.ndarray,
    F_TP: np.ndarray,
    points: np.ndarray,
    pixel_size: float = 0.150,
):
    """Calibrate the Loop-X model.

    Args:
        focal_length: Source to detector distance in mm.
        xs (np.array): (N, M, 2) array of image positions (in Width, Height) for each of the M spheres.
        centers (np.array): (N, 2) array of camera centers in the images.
        theta_S (np.array): (N,) array of source angle measurements, in radians. Note that this is
            0 for an AP shot, so subtract 180 degrees from the machine's reported angle.
        F_TL (np.array): (N, 4, 4) array of poses of the LoopX in the tracker frame.
        F_TP (np.array): (N, 4, 4) array of poses of the pointer in the tracker frame.
        points (np.array): (M, 4) array containing the homogeneous coordinates of the M spheres in the pointer frame.
        pixel_size (float): Size of a pixel in the image, in mm.
    """
    N = xs.shape[0]
    M = xs.shape[1]
    f = focal_length / pixel_size
    ss = np.sin(theta_S)
    cs = np.cos(theta_S)
    rows = []
    for n in range(N):
        u, v = centers[n]
        s, c = ss[n], cs[n]
        l = (inverse(F_TL[n]) @ F_TP[n] @ points.T).T

        fcus = f * c - u * s
        fsuc = -f * s + u * c
        vs = v * s
        vc = v * c
        for m in range(M):
            l0, l1, l2 = l[m, :3] / l[m, 3]
            x0, x1 = xs[n, m]
            # fmt: off
            A = np.array(
                [
                    [s * l0 * x0, 0, c * l0 * x0, s * l1 * x0, 0, c * l1 * x0, s * l2 * x0, 0, c * l2 * x0, s * x0, 0, c * x0, x0],
                    [s * l0 * x1, 0, c * l0 * x1, s * l1 * x1, 0, c * l1 * x1, s * l2 * x1, 0, c * l2 * x1, s * x1, 0, c * x1, x1],
                ]
            )
            B = np.array(
                [
                    [fcus * l0, 0, fsuc * l0, fcus * l1, 0, fsuc * l1, fcus * l2, 0, fsuc * l2, fcus, 0, fsuc, u],
                    [vs * l0, f, vc * l0, vs * l1, f, vc * l1, vs * l2, f, vc * l2, vs, f, vc, v],
                ]
            )
            # fmt: on
            rows.append(A - B)

    A = np.vstack(rows)  # (2NM, 13)
    log.debug(f"A.shape = {A.shape}")
    x, residuals, _, _ = np.linalg.lstsq(A, np.zeros(A.shape[0]))
    log.debug(f"residuals = {residuals}")
    r00, r10, r20, r01, r11, r21, r02, r12, r22, p0, p1, p2, d = x
    # U, S, Vh = np.linalg.svd(A)
    # r00, r10, r20, r01, r11, r21, r02, r12, r22, p0, p1, p2, d = Vh[-1, :]
    R = np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])
    t = np.array([p0, p1, p2])
    F = np.eye(4)
    F[:3, :3] = R
    F[:3, 3] = t
    return F, d


def save_images(
    F_SA_L: np.ndarray,
    d: float,
    focal_length: float,
    xs: np.ndarray,
    centers: np.ndarray,
    theta_S: np.ndarray,
    F_TL: np.ndarray,
    F_TP: np.ndarray,
    points: np.ndarray,
    image_paths: List[Path],
    pixel_size: float = 0.150,
):
    """Save images of the spheres."""
    N = xs.shape[0]
    M = xs.shape[1]
    f = focal_length / pixel_size
    ss = np.sin(theta_S)
    cs = np.cos(theta_S)
    for n in range(N):
        u, v = centers[n]
        s, c = ss[n], cs[n]
        KP = np.array(
            [
                [f, 0, u, 0],
                [0, f, v, 0],
                [0, 0, 1, 0],
            ]
        )
        F_S_SA = np.array(
            [
                [c, 0, -s, 0],
                [0, 1, 0, 0],
                [s, 0, c, d],
                [0, 0, 0, 1],
            ]
        )
        x_hat = (KP @ F_S_SA @ F_SA_L @ inverse(F_TL[n]) @ F_TP[n] @ points.T).T
        x_hat = x_hat[:, :2] / x_hat[:, 2:]

        image = cv2.imread(str(image_paths[n]))
        for m in range(M):
            x, y = x_hat[m]
            log.debug(f"Sphere {m} at ({x}, {y})")
            cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)
        cv2.imwrite(str(Path("images") / f"{image_paths[n].stem}_calib.png"), image)


@click.command()
@click.option("--syncdir", default="~/datasets/OneDrive", help="Sync directory")
@click.option(
    "--plus-dir",
    type=str,
    default="plus-tracking-tron",
    help="Relative path to plus-tracking folder.",
)
@click.option(
    "--images-dir",
    type=str,
    default="datasets/2022-10-30_PointerLoopXCalib2",
    help="Path to images directory: 2022-10-08_PointerLoopXCalibration.",
)
@click.option(
    "--method",
    type=click.Choice(
        [
            "Nelder-Meat",
            "Powell",
            "CG",
            "BFGS",
            "Newton-CG",
            "L-BFGS-B",
            "TNC",
            "COBYLA",
            "SLSQP",
            "trust-constr",
            "dogleg",
            "trust-ncg",
            "trust-exact",
            "trust-krylov",
        ]
    ),
    default="trust-constr",
    help="Optimization method",
)
@click.option(
    "--maxiter",
    type=int,
    default=1000000,
    help="Maximum number of iterations",
)
@click.option(
    "--cache/--no-cache",
    default=True,
    help="Use cached image data",
)
@click.option(
    "--skip/--no-skip",
    default=False,
    help="Skip download",
)
def main(
    syncdir: str, plus_dir: str, images_dir: str, method: str, maxiter: int, cache: bool, skip: bool
):
    onedrive = onedrive_utils.OneDrive(syncdir)
    plus_dir = onedrive.download(plus_dir, skip=skip)
    image_dir = onedrive.download(images_dir, skip=skip)
    data_path = Path("data") / "2022-10-30_loopx_pointer_calibration.csv"

    # TODO: process tracker files instead of cal data

    image_data = load_image_data(data_path, image_dir, plus_dir, cache)

    # loopx is the loopx marker frame
    loopx_from_badgantry = fromstring(
        """-0.0944683	0.232537	0.967989	971.437
        0.995272	0	0.0971309	97.4769
        0.0225865	0.972588	-0.231438	-232.262
        0	0	0	1"""
    )
    badgantry_from_gantry = fromstring(
        """ -0.95403904 0.29096690 0.07174767 -0.00000003
0.27102152 0.93987828 -0.20778878 -0.00000004
-0.12789373 -0.17879346 -0.97553885 -0.00000001
0.00000000 0.00000000 0.00000000 1.00000000"""
    )
    F_L_G = loopx_from_badgantry @ badgantry_from_gantry

    # initial guesses based on early calibrations and documentation
    y_G_SA = 334
    z_G_SA = 506
    F_G_SA = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, y_G_SA],
            [0, 0, 1, z_G_SA],
            [0, 0, 0, 1],
        ]
    )
    ring_from_loopx = inverse(F_G_SA) @ inverse(F_L_G)
    # ring_from_loopx = np.eye(4)
    qx, qy, qz, qw = get_quat(ring_from_loopx)
    p0, p1, p2 = get_pos(ring_from_loopx)
    x0 = np.array([qx, qy, qz, qw, p0, p1, p2])

    # bounds
    bounds = [
        (None, None),  #
        (None, None),  #
        (None, None),  #
        (None, None),  #
        (None, None),  # p0
        (None, None),  # p1
        (None, None),  # p2
    ]
    log.info(f"Bounds: {bounds}")
    spheres_in_index = image_data["points_in_image"].astype(np.float64)
    index_from_isocenter = image_data["index_from_isocenter"].astype(np.float64)
    tracker_from_loopx = image_data["tracker_from_loopx"]
    tracker_from_pointer = image_data["tracker_from_pointer"]
    pixel_size = image_data["pixel_size"].astype(np.float64)
    image_paths = image_data["image_path"]

    # convert spheres_in_pointer to homogeneous coordinates
    spheres_in_pointer_homo = np.concatenate(
        [spheres_in_pointer, np.ones((spheres_in_pointer.shape[0], 1))], axis=1
    )

    # fun = loopx_objective(
    #     x0,
    #     spheres_in_index,
    #     index_from_isocenter,
    #     tracker_from_loopx,
    #     tracker_from_pointer,
    #     spheres_in_pointer_homo,
    #     pixel_size,
    #     verbose=True,
    #     image_paths=image_paths,
    # )

    # # run the optimization
    # res = scipy.optimize.minimize(
    #     value_and_grad(loopx_objective),
    #     x0,
    #     args=(
    #         spheres_in_index,
    #         index_from_isocenter,
    #         tracker_from_loopx,
    #         tracker_from_pointer,
    #         spheres_in_pointer_homo,
    #         pixel_size,
    #     ),
    #     method=method,
    #     bounds=bounds,
    #     jac=True,
    #     options={"maxiter": maxiter},
    #     tol=1e-12,
    # )
    # log.info(f"Optimization result: {res}")

    # # print the results
    # qx, qy, qz, qw, p0, p1, p2 = res.x
    # ring_from_loopx = np.eye(4)
    # ring_from_loopx[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
    # ring_from_loopx[:3, 3] = [p0, p1, p2]

    # fun = loopx_objective(
    #     res.x,
    #     spheres_in_index,
    #     index_from_isocenter,
    #     tracker_from_loopx,
    #     tracker_from_pointer,
    #     spheres_in_pointer_homo,
    #     pixel_size,
    #     verbose=True,
    #     image_paths=image_paths,
    # )
    # log.info(f"Ring from LoopX:\n{tostring(ring_from_loopx)}")

    x = Acquisition.isocenter_from_loopx.as_quatpos()
    fun = loopx_objective(
        x,
        spheres_in_index,
        index_from_isocenter,
        tracker_from_loopx,
        tracker_from_pointer,
        spheres_in_pointer_homo,
        pixel_size,
        verbose=True,
        image_paths=image_paths,
    )


if __name__ == "__main__":
    main()
