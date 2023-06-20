from matplotlib import pyplot as plt
import numpy as np
import pydicom

from cortical_breach_detection.loopx import Acquisition, LoopX
from cortical_breach_detection.utils import onedrive_utils, plus_utils
import logging
from rich.logging import RichHandler
import click
from deepdrr import geo
from deepdrr.utils import dicom_utils, image_utils
import cv2
from pathlib import Path
import pyvista as pv
import scipy

np.set_printoptions(precision=3, suppress=True)


logging.basicConfig(
    level="DEBUG",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger("loopx_calibration")
logging.getLogger("matplotlib").setLevel(logging.WARNING)

# Tracking file: Recording.igs20221028_152628.mha
# Study directory:


def load_fcsv(path: Path) -> np.ndarray:
    """Load a FCSV file from Slicer3D

    Args:
        path (Path): Path to the FCSV file

    Returns:
        np.ndarray: Array of 3D points
    """
    with open(path, "r") as f:
        lines = f.readlines()
    points = []
    for line in lines:
        if line.startswith("#"):
            continue
        x, y, z = line.split(",")[1:4]
        points.append([float(x), float(y), float(z)])
    return np.array(points)


def detect_circles(image: np.ndarray, min_radius: int = 40, max_radius: int = 100) -> np.ndarray:
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
        maxRadius=max_radius,
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


# Always go from front to back. (M,3) array of sphere positions in the pointer frame.   `)
spheres_in_pointer_ = np.array(
    [[43.96, 9.77, 60.70], [-10.76, -1.04, -0.56], [-33.20, -8.73, -60.14]]
)
spheres_in_pointer = [geo.p(x) for x in spheres_in_pointer_]


@click.command()
@click.option("--syncdir", default="~/datasets/OneDrive", help="Sync directory")
@click.option("--skip/--no-skip", default=False, help="Skip existing files")
def main(syncdir: str, skip: bool):
    onedrive = onedrive_utils.OneDrive(syncdir)
    plus_dir = onedrive.download("plus-tracking-tron", skip=skip)
    frames = plus_utils.average_transforms(plus_dir / "data" / "Recording.igs20221104_090503.mha")

    study_dir = onedrive.download(
        "datasets/2022-10-13_corridor-triangulation_phantom/97607477/Study18", skip=skip
    )
    calib2_dir = onedrive.download("datasets/2022-10-30_PointerLoopXCalib2", skip=skip)

    # dicom_path = study_dir / "Series14/2022-11-04.151326834/020727_20221104150026702.dcm" # Image 1 of pelvis
    dicom_path = (
        study_dir / "Series2/2022-11-04.141904733/010201_20221104141738755.dcm"
    )  # Image of reference array.
    slice_path = (
        calib2_dir
        / "20274179/Study6/Series6/2022-10-31.022633977/1.2.826.0.1.3680043.2.940.2022.14.20221031.20426.896.1.0.dcm"
    )
    ds = pydicom.dcmread(slice_path)

    # # Get matrices
    # acquisition = Acquisition(dicom_path)

    # fiducials_path = study_dir / "MarkupsFiducial.fcsv"
    # spheres_in_ct_ = load_fcsv(fiducials_path)
    # spheres_in_ct = [geo.p(x) for x in spheres_in_ct_]

    # image = acquisition.image
    # spheres_in_index = detect_circles(image)
    # log.info(f"Detected {len(spheres_in_index)} spheres in the image:\n{spheres_in_index}")

    # projected_spheres_in_index = [acquisition.index_from_scanner @ x for x in spheres_in_ct]
    # log.info(f"Projected spheres in index:\n{projected_spheres_in_index}")

    # circle_image = image_utils.draw_circles(image, spheres_in_index, color=(0, 0, 255))
    # circle_image = image_utils.draw_circles(
    #     circle_image, projected_spheres_in_index, color=(255, 0, 0)
    # )
    # image_utils.save("images/circles_on_reference.png", circle_image)

    # The transformation matrix from the scanner array on the front side of the scanner to the
    # origin of the coordinate system of the DICOM data (which relates to the ImagingRing coordinate
    # system).
    calibration_matrix_front = np.array(ds[0x30BB2051].value).reshape(4, 4)

    # The geometry of the marker array attached to the front side of the scanner. It allows making
    # sure the calibration and registration are done using the same structure.
    marker_points_front = np.array(ds[0x30BB2052].value).reshape(-1, 3)

    # The transformation matrix from the scanner array on the back side of the scanner to the origin
    # of the coordinate system of the DICOM data (which relates to the ImagingRing coordinate
    # system).
    calibration_matrix_back = np.array(ds[0x30BB2053].value).reshape(4, 4)

    # The geometry of the marker array attached to the back side of the scanner.
    marker_points_back = np.array(ds[0x30BB2054].value).reshape(-1, 3)

    # The transformation matrix from the scanner array to the patient array.
    patient_array_to_scanner_array = np.array(ds[0x30BB2055].value).reshape(4, 4)

    # Start figuring out what things actually are.

    # scanner_from_patient = geo.f(patient_to_scanner)
    # log.debug(f"scanner_from_patient:\n{scanner_from_patient.tostring()}")

    ct_from_scanner_front = geo.f(calibration_matrix_front)
    log.debug(f"ct_from_scanner_front:\n{ct_from_scanner_front.tostring()}")

    markers_in_scanner_front = [geo.p(x) for x in marker_points_front]

    # Which loopx marker corresponds to the scanner marker
    matching = [0, 1, 4, 5, 6, 3, 2]
    markers_in_loopx = [geo.p(LoopX.markers[i]) for i in matching]

    loopx_from_scanner_front = geo.F.from_point_correspondence(
        markers_in_loopx, markers_in_scanner_front
    )

    log.debug(f"markers_in_scanner_front:\n{repr(np.array(markers_in_scanner_front))}")
    log.debug(f"markers_in_loopx:\n{repr(np.array(markers_in_loopx))}")
    log.debug(f"loopx_from_scanner_front:\n{loopx_from_scanner_front.tostring()}")
    exit()

    # plotter = pv.Plotter()
    # colors = ["red", "green", "blue", "yellow", "orange", "purple", "cyan"]
    # for i, m in enumerate(markers_in_loopx):
    #     plotter.add_mesh(pv.Sphere(center=m, radius=5), color="red")
    # for i, m in enumerate(markers_in_scanner_front):
    #     plotter.add_mesh(pv.Sphere(center=loopx_from_scanner_front @ m, radius=5), color="green")
    # plotter.show()

    patient_position = ds.PatientPosition
    tabletop_from_ct = Acquisition.M_PP[patient_position]
    isocenter_from_tabletop = geo.frame_transform(
        """
        -1 0 0 0
        0 -1 0 0
        0 0 1 0
        0 0 0 1
        """
    )
    isocenter_from_ct = isocenter_from_tabletop @ tabletop_from_ct
    isocenter_from_loopx = isocenter_from_ct @ ct_from_scanner_front @ loopx_from_scanner_front.inv
    log.debug(f"CT-based isocenter_from_loopx:\n{isocenter_from_loopx.tostring()}")

    # Patient is the reference marker, but in their geometry.
    scanner_array_from_patient_array = geo.f(patient_array_to_scanner_array)
    log.debug(f"scanner_from_patient:\n{scanner_array_from_patient_array.tostring()}")
    # Should be which array was used.
    # log.debug(ds[0x30BB2057].value)
    scanner_from_reference = (
        loopx_from_scanner_front.inv
        @ frames[("tracker", "loop_x")].inv
        @ frames[("tracker", "reference")]
    )
    log.debug(f"scanner_from_reference:\n{scanner_from_reference.tostring()}")

    exit()

    """
    Single Image Debug
    """
    image_path = study_dir / "Series1" / "2022-10-28.213404444" / "009506_20221028212446299.dcm"
    image = dicom_utils.read_image(image_path)
    spheres_in_index_ = detect_circles(image, min_radius=40, max_radius=80)

    # Sort the points in the image by their x coordinate, in descending order,
    # so that the points are front-to-back. (pointer was oriented same way in all images.)
    spheres_in_index_ = spheres_in_index_[np.argsort(spheres_in_index_[:, 0])[::-1]]

    spheres_in_index = [geo.p(x) for x in spheres_in_index_]

    circle_image = image_utils.ensure_cdim(image.copy())
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
    ]
    for i, (x, y) in enumerate(spheres_in_index_):
        cv2.circle(circle_image, (x, y), 10, colors[i], 10)
    image_utils.save("images/circles_test.png", circle_image)
    fiducials_path = study_dir / "MarkupsFiducial.fcsv"
    spheres_in_ct_ = load_fcsv(fiducials_path)[::-1]  # Labeled in reverse order
    spheres_in_ct = [geo.p(x) for x in spheres_in_ct_]

    # Read the frames for this.
    mha_path = plus_dir / "data" / "Recording.igs20221028_152628.mha"
    frames = plus_utils.average_transforms(mha_path)

    ct_from_pointer = geo.F.from_point_correspondence(spheres_in_ct, spheres_in_pointer)
    pointer_from_loopx = frames[("tracker", "pointer")].inv @ frames[("tracker", "loop_x")]
    ct_from_loopx = ct_from_pointer @ pointer_from_loopx

    acquisition = Acquisition(image_path, ct_from_loopx)

    log.debug(f"acquisition orientation: {acquisition.patient_position}")

    P = acquisition.projection_matrix[[0, 1, 3], :]
    index_from_ct = acquisition.index_from_patient
    log.debug(f"index from ct: {index_from_ct.data}")

    # TODO: figure this the fuck out.
    for i, m in enumerate(spheres_in_ct):
        log.debug(f"Sphere {i} in ct: {m}")
        p = P @ m.get_data()
        p = p / p[2]
        log.debug(f"In detector mm: {p}")
        p = p / 0.15
        log.debug(f"In index (on P): {p}, gt: {spheres_in_index_[i]}")

        u_hat = index_from_ct @ m
        log.debug(f"Sphere {i} in camera: {acquisition.camera_from_patient @ m}")
        log.info(f"Sphere {i}: {u_hat}, gt: {spheres_in_index[i]}")

    isocenter_from_loopx = acquisition.isocenter_from_patient @ ct_from_loopx
    log.debug(f"Isocenter from loopx:\n{isocenter_from_loopx.tostring()}")


if __name__ == "__main__":
    main()
