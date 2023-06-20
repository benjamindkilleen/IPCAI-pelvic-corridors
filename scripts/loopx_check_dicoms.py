import pydicom
import csv
import numpy as np
from pathlib import Path
import click
from rich.logging import RichHandler
import logging
from cortical_breach_detection.utils.onedrive_utils import OneDrive
from typing import Dict, Any
import csv
import pyvista as pv
import matplotlib.pyplot as plt
from deepdrr import geo

logging.basicConfig(
    level="DEBUG",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger("script")
np.set_printoptions(precision=3, suppress=True)


def get_principle_ray(source_angle, detector_angle):
    theta_s = np.radians(source_angle)
    theta_d = np.radians(detector_angle)
    s = np.array([np.sin(theta_s), 0, np.cos(theta_s)])
    d = np.array([np.sin(theta_d), 0, np.cos(theta_d)])
    r = d - s
    return r / np.linalg.norm(r)


def get_camera_projection(dicom_path: Path) -> geo:
    ds = pydicom.dcmread(dicom_path)
    H = ds[0x00280010].value
    W = ds[0x00280011].value
    pixel_size = ds[0x00280030].value
    shutter_vertices = np.array(ds[0x00181620].value).reshape(-1, 2)


def get_loopx_params(dicom_path: Path) -> Dict[str, Any]:
    """Get the LoopX parameters from a DICOM file.

    - 0x30bb2080: source position in the some frame, e.g. "[168.948954003302, 0.8259713206867916, -728.0712691609561]" for an AP shot.
    - 0x30BB2084: source angle
    - 0x30BB2086: detector angle
    - 0x30BB2088: gantry tilt

    """
    # Get the known parameters of the loop-x from the image. This does not seemingly include things
    # like longitudinal, lateral, or traction yaw, which relative to the home position. More work
    # would be needed. But this is good for now.
    ds = pydicom.dcmread(dicom_path)
    source_angle = float(ds[0x30BB2084].value)
    detector_angle = float(ds[0x30BB2086].value)
    gantry_tilt = float(ds[0x30BB2088].value)
    return dict(
        source_angle=source_angle,
        detector_angle=detector_angle,
        gantry_tilt=gantry_tilt,
    )


@click.command()
@click.option("--syncdir", type=str, default="~/datasets/OneDrive")
@click.option("--remote-dir", type=str, default="datasets/2022-10-26_DicomNavTest")
def main(syncdir: str, remote_dir: str):
    onedrive = OneDrive(syncdir)
    onedrive.download(remote_dir)

    local_dir = onedrive.syncdir / remote_dir
    file = open(local_dir / "loopx.csv", "w")
    writer = csv.writer(file)
    writer.writerow("filename,source_angle,detector_angle,gantry_tilt,s0,s1,s2".split(","))

    points = []
    labels = []
    source_points = []
    detector_points = []
    detector_distances = []
    source_to_detector_distances = []
    for i, path in enumerate(sorted(list(local_dir.glob("**/*.dcm")))):

        log.info("----------------")
        ds = pydicom.dcmread(path)
        source_angle = float(ds[0x30BB2084].value)
        detector_angle = float(ds[0x30BB2086].value)
        gantry_tilt = float(ds[0x30BB2088].value)
        label = f"sdg: {source_angle:.01f}, {detector_angle:.01f}, {gantry_tilt:.01f}"
        log.info(label)
        s = np.array(ds[0x30BB2080].value)
        d = np.array(ds[0x30BB2081].value)

        # Unit vectors pointing in direction of the indexing for the images.
        row_direction = np.array(ds[0x30BB2082].value)
        col_direction = np.array(ds[0x30BB2083].value)

        # Only question is whether d is at the image edge after cropping or before.

        log.info(f"source position: {s}")
        log.info(f"detector position: {d}")
        p = d + 2880 * 0.150 * row_direction
        q = d + 2880 * 0.150 * col_direction
        log.debug(f"p: {p}")

        # r = -np.cross(ray1, ray2)  # negate to point toward detector.
        # r = 1500 * r / np.linalg.norm(r)
        r = 1500 * get_principle_ray(source_angle, detector_angle)

        shutter_vertices = np.array(ds[0x00181620].value).reshape(-1, 2)

        image_origin: geo.Point3D = geo.p(d)
        n: geo.Vector3D = geo.v(row_direction).cross(geo.v(col_direction))
        image_plane = geo.plane(image_origin, n)
        detector_distances.append(image_plane.distance(geo.p(0, 0, 0)))
        source_to_detector_distances.append(image_plane.distance(geo.p(s)))

        # abs((source_angle % 360) - 180) > 5:
        if path.stem in ["002812_20221026231933786", "002942_20221026232017585"]:
            origin = pv.Sphere(radius=5, center=[0, 0, 0])
            origin += pv.Line([0, 0, 0], [0, 0, 100])
            origin += pv.Line([0, 0, 0], [0, 100, 0])
            origin += pv.Line([0, 0, 0], [100, 0, 0])

            # log.debug(f"ds: {ds}")
            verts = pv.PolyData()
            for vert in shutter_vertices:
                center = d + vert[0] * 0.150 * row_direction + vert[1] * 0.150 * col_direction
                verts += pv.Sphere(center=center, radius=10)
            log.debug(f"shutter_vertices:\n{shutter_vertices}")
            source = pv.Sphere(center=s, radius=10)
            source += pv.Line(s, s + r)
            detector = pv.Sphere(center=d, radius=10)
            r1 = pv.Line(d, d + 2880 * 0.150 * row_direction)
            r2 = pv.Line(d, d + 2880 * 0.150 * col_direction)

            # plotter = pv.Plotter()
            # plotter.add_mesh(origin, color="black")
            # plotter.add_mesh(verts, color="magenta")
            # plotter.add_title(label)
            # plotter.add_mesh(source, color="g")
            # plotter.add_mesh(detector, color="r")
            # plotter.add_mesh(r1, color="red")
            # plotter.add_mesh(r2, color="blue")
            # plotter.show()

        points.append(row_direction)
        labels.append(label)
        source_points.append(s)
        detector_points.append(d)

    file.close()

    avg_source_distance = np.mean(np.linalg.norm(source_points, axis=1))
    std_source_distance = np.std(np.linalg.norm(source_points, axis=1))

    log.info(f"source distance: {avg_source_distance:.02f} +/- {std_source_distance:.02f}")
    log.info(
        f"detector plane distances: {np.mean(detector_distances):.02f} +/- {np.std(detector_distances):.02f}"
    )
    log.info(
        f"source to detector distances: {np.mean(source_to_detector_distances):.02f} +/- {np.std(source_to_detector_distances):.02f}"
    )


if __name__ == "__main__":
    main()
