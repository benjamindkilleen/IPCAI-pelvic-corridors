from functools import partial
from logging.config import fileConfig
import math
from datetime import datetime
from pathlib import Path
from typing import List
import numpy as np
import scipy.optimize
from scipy.spatial.transform import Rotation as R
import click
from deepdrr.utils import data_utils, image_utils
import logging
from rich.logging import RichHandler
import matplotlib.pyplot as plt
from rich.progress import track
from pycpd import RigidRegistration
import pyvista as pv
import trimesh
import trimesh.registration
from deepdrr import geo


logging.basicConfig(
    level="DEBUG",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger("loopx_calibration")
logging.getLogger("matplotlib").setLevel(logging.WARNING)


def fromstring(s: str) -> np.ndarray:
    """Convert a string representation of a 4x4 matrix into a numpy array."""
    return np.array(s.split()).astype(float).reshape(4, 4)


def fromtxt(path: Path) -> np.ndarray:
    """Load a txt file with a transformation matrix."""
    with open(path, "r") as f:
        lines = f.readlines()
    l = None
    for line in lines:
        if line.startswith("Parameters:"):
            l = np.fromstring(line.split(":")[1], sep=" ", dtype=float)
            break

    if l is None:
        raise ValueError("Could not find Parameters in file")

    r = l[:9].reshape(3, 3).T
    t = l[9:12]
    return np.vstack((np.hstack((r, t[:, np.newaxis])), np.array([0, 0, 0, 1])))


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


def load_csv(path: Path) -> np.ndarray:
    """Load a CSV file from Slicer3D

    Args:
        path (Path): Path to the CSV file
    """
    with open(path, "r") as f:
        lines = f.readlines()
    points = []
    for line in lines[1:]:
        if line.startswith("#"):
            continue
        x, y, z = line.split(",")[1:4]
        points.append([float(x), float(y), float(z)])
    return np.array(points)


def tostring(A: np.ndarray) -> str:
    """Convert a 4x4 matrix into a string representation with newlines."""
    formatter = {"float_kind": lambda x: "%.8f" % x}
    lines = [
        np.array2string(A[i], separator=" ", formatter=formatter)[1:-1] for i in range(A.shape[0])
    ]
    return "\n".join(lines)


LPS_from_RAS = np.array(
    [
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
)


@click.command()
@click.option(
    "--data-dir",
    type=click.Path(exists=True, file_okay=False),
    default="/home/killeen/datasets/OneDrive/datasets/2022-07_AR-Fluoroscopy/2022-08-17_LFOV-CT",
    help="2022-08-17_LFOV-CT directory containing surface models, etc.",
)
def main(data_dir: str):
    data_dir = Path(data_dir).expanduser()

    # Load the phantom model
    model_path = data_dir / "PelvisPhantomModel.stl"
    pelvis_model = trimesh.load(model_path)
    mesh = pv.read(model_path)

    # Maybe in RAS?
    fiducials_pelvismodel = load_fcsv(data_dir / "fiducials_PelvisModel.fcsv")
    fiducials_pelvismarker = load_fcsv(data_dir / "fiducials_PelvisMarker.fcsv")

    # Load the collected_points in the marker frame
    collected_points_path = data_dir / "SurPoints_PelvisMarker.fcsv"
    points_pelvismarker = load_fcsv(collected_points_path)

    # Slicer got the following:
    # -0.921585 -0.376626 -0.0939832 22.1822
    # -0.160806 0.150054 0.975513 -37.0712
    # -0.353301 0.914131 -0.198852 -153.747
    # 0 0 0 1

    model_from_marker_init = np.array(
        geo.FrameTransform.from_point_correspondence(fiducials_pelvismodel, fiducials_pelvismarker)
    )
    points_pelvismodel_init = trimesh.transform_points(points_pelvismarker, model_from_marker_init)
    log.info(f"Initial model_from_marker:\n{tostring(model_from_marker_init)}")
    model_from_marker_init_RAS = LPS_from_RAS @ model_from_marker_init @ LPS_from_RAS
    log.info(f"Initial model_from_marker_RAS:\n{tostring(model_from_marker_init_RAS)}")

    # ps_pelvismarker = pv.PolyData(points_pelvismarker)
    # ps_pelvismodel = pv.PolyData(points_pelvismodel_init)
    # fids_pelvismarker = pv.PolyData(points_pelvismarker[points_indices])
    # fids_pelvismodel = pv.PolyData(fiducials_pelvismodel[fiducials_indices])
    # plotter = pv.Plotter()
    # plotter.add_mesh(ps_pelvismarker, color="red", point_size=2)
    # plotter.add_mesh(ps_pelvismodel, color="green", point_size=2)
    # plotter.add_mesh(mesh, color="white")
    # plotter.show()
    # exit()

    # Run icp
    model_from_marker, points_pelvismodel, cost = trimesh.registration.icp(
        points_pelvismarker, pelvis_model, initial=model_from_marker_init
    )
    log.info(f"ICP succeeded with cost {cost}")

    # ps_pelvismodel = pv.PolyData(points_pelvismodel)
    # plotter = pv.Plotter()
    # plotter.add_mesh(ps_pelvismodel, color="red", point_size=3)
    # plotter.add_mesh(mesh, color="white")
    # plotter.show()

    # Save the transformation
    log.info(f"model_from_marker:\n{tostring(model_from_marker)}")

    model_from_marker_RAS = LPS_from_RAS @ model_from_marker @ LPS_from_RAS
    log.info(f"model_from_marker_RAS):\n{tostring(model_from_marker_RAS)}")


if __name__ == "__main__":
    main()
