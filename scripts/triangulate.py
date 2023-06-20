from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
import logging

import h5py
import numpy as np
from deepdrr import geo
from rich.logging import RichHandler

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, handlers=[RichHandler()])


def triangulate(
    points: List[geo.Point2D], camera3d_from_worlds: List[geo.FrameTransform], index_from_camera3d: geo.Transform
) -> geo.Point3D:
    """

    Based on Section 7.1 in Szeliski 2010 Computer Vision and Applications, which is in turn based on Sutherland, 1974

    TODO: allow for a list of points

    Args:
        points: image-space points (in the deepdrr convention, with [column, row, 1]) corresponding to the same 3D point in each image.
        camera3d_from_worlds: List of camera poses for each image (i.e. the extrinsic matrices [R|t]).
        index_from_camera3d: The product of the intrinsic calibration matrix K and the projection model matrix.

    """

    if len(points) != len(camera3d_from_worlds):
        raise ValueError("Must provide one point for every image.")

    # [N, 3, 4] array of camera projections matrices
    A = []
    for j in range(len(points)):
        x, y = points[j]
        P = index_from_camera3d @ camera3d_from_worlds[j]
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])

    A = np.array(A)

    p, _, _, _ = np.linalg.lstsq(A, np.zeros(A.shape[0]))
    return geo.Point3D(p)  # divides by w


def slicer_to_itk(vec: np.ndarray):
    ITK_xform = np.concatenate([vec.reshape(4, 3).T, [[0, 0, 0, 1]]], axis=0)

    RAS2LPS = np.array(
        [
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    ITK_xform = RAS2LPS @ ITK_xform @ RAS2LPS
    tmp1 = ITK_xform[0, 0] * ITK_xform[0, 3] + ITK_xform[0, 1] * ITK_xform[1, 3] + ITK_xform[0, 2] * ITK_xform[2, 3]
    tmp2 = ITK_xform[1, 0] * ITK_xform[0, 3] + ITK_xform[1, 1] * ITK_xform[1, 3] + ITK_xform[1, 2] * ITK_xform[2, 3]
    tmp3 = ITK_xform[2, 0] * ITK_xform[0, 3] + ITK_xform[2, 1] * ITK_xform[1, 3] + ITK_xform[2, 2] * ITK_xform[2, 3]

    ITK_xform[0, 3] = -tmp1
    ITK_xform[1, 3] = -tmp2
    ITK_xform[2, 3] = -tmp3
    return ITK_xform


def read_itk_affine_transform_from_file(path: str) -> geo.FrameTransform:
    f = h5py.File(path)
    transform_group = f["TransformGroup"]["0"]
    itk_xform = np.array(transform_group["TranformParameters"])
    itk_xform = itk_xform.reshape(3, 4)
    # TODO: not sure this is the right way to load these.
    return geo.FrameTransform.from_rt(rotation=itk_xform[:, :3], translation=itk_xform[:, 3])


def main():
    root = Path("~/datasets").expanduser()
    kwire_exp_dir = root / "2022-02_Bayview_Cadaver" / "KwireExp_Benjamin"
    handeyeX_path = kwire_exp_dir / "handeye_result" / "handeye_X.h5"
    handeyeX_transform = read_itk_affine_transform_from_file(handeyeX_path)

    calibration_data_dir = kwire_exp_dir / "calibration_data"
    frame_dirs = sorted([d for d in calibration_data_dir.iterdir() if d.is_dir()])

    carm_fiducial_transforms = []
    camera3d_from_worlds = []  # "world" is the first camera frame
    for i, frame_dir in enumerate(frame_dirs):
        carm_transform_path = frame_dir / "BayviewSiemensCArmInverse.h5"
        f = h5py.File(carm_transform_path, "r")
        slicer_transform = np.array(f["TransformGroup"]["0"]["TranformParameters"])
        itk_transform = geo.FrameTransform(slicer_to_itk(slicer_transform))
        carm_fiducial_transforms.append(itk_transform)
        camera3d_from_world = (
            handeyeX_transform.inv @ carm_fiducial_transforms[i].inv @ carm_fiducial_transforms[0] @ handeyeX_transform
        )
        camera3d_from_worlds.append(camera3d_from_world)

    # looking at that upper left fiducial in the first three images
    points = [
        [154.8, 117.9],
        [143.9, 114.27],
        [142, 112.5],
    ]

    intrinsic = geo.CameraIntrinsicTransform(np.array([[-5257.73, 0, 767.5], [0, -5257.73, 767.5], [0, 0, 1]]))
    proj = geo.CameraProjection(intrinsic, extrinsic=camera3d_from_worlds[0])

    p = triangulate(
        points=points,
        camera3d_from_worlds=camera3d_from_worlds[: len(points)],
        index_from_camera3d=proj.index_from_camera3d,
    )
    log.info(f"reconstructed point: {p}")


if __name__ == "__main__":
    main()
