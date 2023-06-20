from matplotlib import pyplot as plt
import numpy as np
import pydicom

from typing import List
from cortical_breach_detection.loopx import Acquisition, LoopX
from cortical_breach_detection.utils import onedrive_utils, plus_utils, cv_utils
import logging
from rich.logging import RichHandler
import click
from deepdrr import geo
from deepdrr.annotations import FiducialList
from deepdrr.utils import dicom_utils, image_utils
import cv2
from pathlib import Path
import pyvista as pv
import scipy
from scipy.spatial.distance import cdist
import pyransac3d as pyrsc

np.set_printoptions(precision=3, suppress=True)


logging.basicConfig(
    level="DEBUG",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger("loopx_calibration")
logging.getLogger("matplotlib").setLevel(logging.WARNING)

points_in_reference_ = np.fromstring(
    """67.56 48.73 -20.11
52.62 -67.19 -24.84
-44.82 76.59 22.78
-75.36 -58.13 22.17""",
    sep=" ",
).reshape(-1, 3)

points_in_reference: List[geo.Point3D] = [geo.point(p) for p in points_in_reference_]


def fit_sphere(points: np.ndarray) -> geo.Point3D:
    sphere = pyrsc.Sphere()
    center, radius, inliers = sphere.fit(points, thresh=0.1)
    log.debug(f"center: {center}, radius: {radius}")
    return geo.point(center)


def load_fids(path: Path) -> FiducialList:
    points = []
    for p in sorted(list(path.glob("F*.stl"))):
        points.append(fit_sphere(pv.read(p).points))

    return FiducialList(points, anatomical_coordinate_system="LPS")


@click.command()
@click.option("--syncdir", default="~/datasets/OneDrive", help="Sync directory")
@click.option("--skip/--no-skip", default=False, help="Skip existing files")
def main(syncdir: str, skip: bool):
    onedrive = onedrive_utils.OneDrive(syncdir)

    data_dir = onedrive.download("datasets/2022-11-05_patient_array", skip=skip)

    log.debug(f"Loading fiducials...")
    points_in_LPS = FiducialList.from_fcsv(data_dir / "PatientArrayMarkers_LPS.fcsv")

    acquisition = Acquisition(
        data_dir / "87068508/Study3/Series1/2022-11-05.210057434/002579_20221105205502682.dcm"
    )

    LPS_from_reference = geo.FrameTransform.from_points(points_in_LPS, points_in_reference)
    patient_array_from_reference = (
        acquisition.patient_array_from_scanner @ acquisition.scanner_from_LPS @ LPS_from_reference
    )
    points_in_patient_array = patient_array_from_reference @ points_in_reference
    log.info(f"points_in_patient_array:\n{repr(np.array(points_in_patient_array))}")
    log.debug(f"points in patient array mean:\n{np.mean(points_in_patient_array, axis=0)}")
    log.info(f"Patient array from reference:\n{patient_array_from_reference.tostring()}")

    # log.info(f"Scanner from reference:\n{scanner_from_reference.tostring()}")
    errors = [
        (p_LPS - LPS_from_reference @ p_ref).norm()
        for p_LPS, p_ref in zip(points_in_LPS, points_in_reference)
    ]
    log.info(f"Transform error: {np.mean(errors):.2f} ± {np.std(errors):.2f} mm")

    # Reproject onto the acquisition plane
    points_in_index = acquisition.index_from_scanner @ list(points_in_LPS)

    image = acquisition.image
    circles = cv_utils.detect_circles(image, 40, 100, 20)
    circles = circles[:4]
    # log.info(f"Found {len(circles)} circles:\n{circles}")
    gt_points_in_index = circles[:, :2]
    projection_errors = cdist(np.array(points_in_index), gt_points_in_index).min(axis=1)
    log.info(
        f"Projection error: {np.mean(projection_errors):.2f} ± {np.std(projection_errors):.2f} pixels"
    )
    projection_errors_mm = projection_errors * acquisition.pixel_size[0]
    log.info(
        f"Projection error: {np.mean(projection_errors_mm):.2f} ± {np.std(projection_errors_mm):.2f} mm"
    )
    image = image_utils.draw_circles(image, circles, (0, 255, 0), 2, radius=20)
    image = image_utils.draw_circles(image, points_in_index, (255, 0, 0), 2, radius=20)
    image_utils.save("images/circles_patient_array.png", image)


if __name__ == "__main__":
    main()
