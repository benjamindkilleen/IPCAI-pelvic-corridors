import csv
import math
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Optional, Union
from cv2 import setRNGSeed
import logging

import h5py as h5
import nibabel as nib
import torch
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from skimage.exposure import match_histograms
import torchvision.transforms.functional as TF

from torchvision.transforms import InterpolationMode
import pytorch_lightning as pl
import deepdrr
from deepdrr import geo
from deepdrr.load_dicom import conv_hu_to_density
from deepdrr.utils import image_utils
from deepdrr.annotations import LineAnnotation
from deepdrr.vis import get_frustum_mesh
import pyvista as pv
from rich.progress import track

import numpy as np

import PIL
from PIL import Image


from .models import UNet
from .utils.onedrive_utils import OneDrive
from . import utils
from .utils import geo_utils

log = logging.getLogger(__name__)

# 1: left hip
# 2: right hip
# 3: vertebrae
# 4: sacrum
# 5: left femur
# 6: right femur
# 7: tailbone
# 8: vertebra


def get_detector_plane(
    camera_projection: geo.CameraProjection,
    pixel_size: float,
) -> geo.Plane:
    focal_length_mm = camera_projection.intrinsic.focal_length * pixel_size
    p = geo.plane(geo.p(0, 0, focal_length_mm), geo.v(0, 0, 1))
    p = camera_projection.world_from_camera3d @ p
    return p


def isin_image(point: geo.Point2D, H: int, W: int, pad: int = 20) -> bool:
    return point.x >= pad and point.x < W - pad and point.y >= pad and point.y < H - pad


def triangulate_robdata(
    model: UNet,
    onedrive_dir: str = "~/datasets/OneDrive",
    specimen_groups: List[str] = ["17-1882", "17-1905", "18-0725", "18-1109", "18-2799", "18-2800"],
    pad: int = 12,
    image_size: List[int] = [256, 256],
    check_drr: bool = False,
):
    """Go through and get all the images, as well as the projection matrices."""
    onedrive = OneDrive(onedrive_dir)
    data_dir = onedrive.download("datasets/DeepFluoro", skip=True)
    h5_path = Path(data_dir) / "ipcai_2020_full_res_data.h5"

    f = h5.File(str(h5_path), "r")

    proj_group = f["proj-params"]
    log.debug(f"proj_group: {proj_group.keys()}")

    robcam_from_robworld = geo.F(np.array(proj_group["extrinsic"]))
    H = int(proj_group["num-rows"][()])
    W = int(proj_group["num-cols"][()])
    log.debug(f"sensor_height: {H}")
    log.debug(f"sensor_width: {W}")
    pixel_size = float(proj_group["pixel-col-spacing"][()])
    intrinsic = geo.CameraIntrinsicTransform(
        np.array(proj_group["intrinsic"]),
        sensor_height=H,
        sensor_width=W,
    )

    cam_from_robcam = geo.f(
        """
        -1 0 0 0
        0 -1 0 0
        0 0 -1 0
        0 0 0 1
        """
    )

    log.debug(f"robcam_from_world: {robcam_from_robworld}")
    log.debug(f"intrinsic: {intrinsic}")

    results_file = open(f"results.csv", "w+")
    fieldnames = [
        "specimen",
        "side",
        "image_1",
        "image_2",
        "relative_rotation_deg",
        "rotation_about_corridor_deg",
        "corridor_startpoint_error_mm",
        "corridor_angle_error_deg",
        "corridor_circumscribing_radius_mm",
        "cortical_breach_pred",
        "triangulation_time",
    ]
    csv_writer = csv.DictWriter(results_file, fieldnames=fieldnames)
    csv_writer.writeheader()

    for specimen_name in specimen_groups:
        log.info(f"Processing specimen: {specimen_name}")
        specimen_group = f[specimen_name]

        volume_group = specimen_group["vol"]
        hu_volume = np.array(volume_group["pixels"]).astype(np.float32)
        hu_volume = np.moveaxis(hu_volume, [0, 1, 2], [2, 1, 0]).copy()
        data = conv_hu_to_density(hu_volume)
        origin = np.array(volume_group["origin"]).squeeze()
        spacing = np.array(volume_group["spacing"]).squeeze()
        rotation_mat = np.array(volume_group["dir-mat"])

        seg_group = specimen_group["vol-seg"]["image"]
        seg = np.array(seg_group["pixels"]).astype(np.float32)
        seg = np.moveaxis(seg, [0, 1, 2], [2, 1, 0]).copy()
        seg_origin = np.array(seg_group["origin"]).squeeze()
        seg_spacing = np.array(seg_group["spacing"]).squeeze()
        seg_rotation_mat = np.array(seg_group["dir-mat"])

        materials = dict(bone=seg > 0, air=seg == 0)

        log.debug(f"rotation_mat: {rotation_mat}")
        anatomical_from_ijk = geo.FrameTransform.from_rt(rotation_mat, origin)
        ct = deepdrr.Volume(
            data,
            materials=materials,
            anatomical_from_ijk=anatomical_from_ijk,
            world_from_anatomical=None,
            anatomical_coordinate_system="LPS",
        )
        pelvis_vol = deepdrr.Volume(
            np.isin(seg, [1, 2, 3, 4, 7]).astype(np.float32),
            materials=materials,
            anatomical_from_ijk=anatomical_from_ijk,
            world_from_anatomical=None,
            anatomical_coordinate_system="LPS",
        )

        # Save the pelvis as a nifti
        img_path = data_dir / f"{specimen_name}_pelvis.nii.gz"
        seg_path = data_dir / f"{specimen_name}_pelvis_seg.nii.gz"
        # NOTE: These are saved out in LPS, so we need to flip them to RAS
        if not img_path.exists():
            log.info(f"Saving {img_path}")
            img = nib.Nifti1Image(hu_volume, np.array(geo.RAS_from_LPS @ ct.anatomical_from_ijk))
            nib.save(img, img_path)
        if not seg_path.exists():
            log.info(f"Saving {seg_path}")
            seg_img = nib.Nifti1Image(seg, np.array(geo.RAS_from_LPS @ ct.anatomical_from_ijk))
            nib.save(seg_img, seg_path)

        surface_path = data_dir / f"{specimen_name}_pelvis_surface.stl"
        if not surface_path.exists():
            surface = pelvis_vol.isosurface()
            surface.save(surface_path)
        else:
            surface = pv.read(surface_path)

        projector = deepdrr.Projector(
            ct,
            camera_intrinsics=intrinsic,
            intensity_upper_bound=10,
        )
        if check_drr:
            projector.initialize()

        # Ground truth annotations.
        left_annotation = LineAnnotation.from_json(
            data_dir / f"{specimen_name}_left.mrk.json", volume=ct
        )
        right_annotation = LineAnnotation.from_json(
            data_dir / f"{specimen_name}_right.mrk.json", volume=ct
        )

        # List of principle rays for all the images, in LPS coordinates
        principle_rays: List[geo.Vector3D] = []
        sources: List[geo.Point3D] = []
        camera_projections: List[geo.CameraProjection] = []
        images = []
        left_corridor_isin_image = []
        right_corridor_isin_image = []
        projection_keys = []
        description = f"Getting projections for {specimen_name}:"
        for proj_key in track(list(specimen_group["projections"].keys()), description=description):
            projection_group = specimen_group["projections"][proj_key]
            image = np.array(projection_group["image"]["pixels"]).astype(np.float32)
            image[pad : image.shape[0] - pad, pad : image.shape[1] - pad] = deepdrr.utils.neglog(
                image[pad : image.shape[0] - pad, pad : image.shape[1] - pad]
            )
            image[:pad, :] = 0
            image[image.shape[0] - pad :, :] = 0
            image[:, :pad] = 0
            image[:, image.shape[1] - pad :] = 0

            pelvis_vol_from_robcam = geo.F(
                np.array(projection_group["gt-poses"]["cam-to-pelvis-vol"])
            )

            robcam_from_pelvis_vol = robcam_from_robworld @ pelvis_vol_from_robcam.inverse()

            # "world" is just LPS
            index_from_world = geo.CameraProjection(
                intrinsic=intrinsic,
                extrinsic=cam_from_robcam @ robcam_from_pelvis_vol,
            )

            if projection_group["rot-180-for-up"][()]:
                image = np.rot90(image, k=2, axes=(0, 1)).copy()
                index_from_world = (
                    geo.F(
                        np.array(
                            [
                                [-1, 0, W],
                                [0, -1, H],
                                [0, 0, 1],
                            ]
                        )
                    )
                    @ index_from_world
                )

            left_startpoint_in_index = index_from_world @ left_annotation.startpoint_in_world
            left_endpoint_in_index = index_from_world @ left_annotation.endpoint_in_world
            right_startpoint_in_index = index_from_world @ right_annotation.startpoint_in_world
            right_endpoint_in_index = index_from_world @ right_annotation.endpoint_in_world

            # Append to lists.
            projection_keys.append(proj_key)
            images.append(image)
            camera_projections.append(index_from_world)
            principle_rays.append(index_from_world.principle_ray_in_world)
            sources.append(index_from_world.center_in_world)

            if isin_image(left_startpoint_in_index, H, W, pad=40) and isin_image(
                left_endpoint_in_index, H, W, pad=40
            ):
                left_corridor_isin_image.append(True)
            else:
                left_corridor_isin_image.append(False)

            if isin_image(right_startpoint_in_index, H, W, pad=40) and isin_image(
                right_endpoint_in_index, H, W, pad=40
            ):
                right_corridor_isin_image.append(True)
            else:
                right_corridor_isin_image.append(False)

            if check_drr:
                drr = projector(index_from_world)
                left_circles = np.array(
                    [
                        left_startpoint_in_index,
                        left_endpoint_in_index,
                    ]
                )
                right_circles = np.array(
                    [
                        right_startpoint_in_index,
                        right_endpoint_in_index,
                    ]
                )
                image = image_utils.draw_circles(image, left_circles, radius=20, color=(255, 0, 0))
                image = image_utils.draw_circles(image, right_circles, radius=20, color=(0, 255, 0))
                image_utils.save(f"images/{specimen_name}_{proj_key}_image.png", image)
                image_utils.save(f"images/{specimen_name}_{proj_key}_drr.png", drr)

        projector.free()
        images = np.array(images)
        left_corridor_isin_image = np.array(left_corridor_isin_image)
        right_corridor_isin_image = np.array(right_corridor_isin_image)

        for left_side in [True, False]:
            if left_side:
                corridor_isin_image = left_corridor_isin_image
                annotation = left_annotation
                annotation_dir = Path(f"{specimen_name}_left_corridors")
            else:
                corridor_isin_image = right_corridor_isin_image
                annotation = right_annotation
                annotation_dir = Path(f"{specimen_name}_right_corridors")

            annotation_dir.mkdir()

            visited = set()
            gt_direction = annotation.direction_in_world
            side = "left" if left_side else "right"
            description = f"Getting corridors for {specimen_name} {side}:"
            for idx_1 in track(np.argwhere(corridor_isin_image).flatten(), description=description):
                for idx_2 in np.argwhere(corridor_isin_image).flatten():
                    results = {}

                    if idx_1 == idx_2 or (idx_1, idx_2) in visited:
                        continue
                    visited.add((idx_1, idx_2))
                    visited.add((idx_2, idx_1))

                    # Get the principle rays
                    ray_1 = principle_rays[idx_1]
                    ray_2 = principle_rays[idx_2]

                    rotvec = ray_1.rotvec_to(ray_2)
                    relative_rotation = np.linalg.norm(rotvec)
                    if relative_rotation < np.radians(5):
                        continue

                    rotation_about_corridor_deg = abs(math.degrees(rotvec.dot(gt_direction)))
                    if rotation_about_corridor_deg < 30:  # 15:
                        continue

                    proj_key_1 = projection_keys[idx_1]
                    proj_key_2 = projection_keys[idx_2]
                    image_1 = images[idx_1]
                    image_2 = images[idx_2]
                    index_from_world_1 = camera_projections[idx_1]
                    index_from_world_2 = camera_projections[idx_2]

                    results["specimen"] = specimen_name
                    results["image_1"] = proj_key_1
                    results["image_2"] = proj_key_2
                    results["side"] = side

                    # Triangulate the corridor
                    t = time.time()
                    startpoint, direction, _, _ = model.triangulate(
                        image_1=image_1,
                        image_2=image_2,
                        index_from_world_1=index_from_world_1,
                        index_from_world_2=index_from_world_2,
                        left_side=left_side,
                        image_size=image_size,
                    )
                    results["triangulation_time"] = time.time() - t

                    if startpoint is None or direction is None:
                        continue

                    corridor = geo.line(startpoint, direction)
                    startpoint_error_mm = corridor.distance(annotation.startpoint_in_world)
                    direction_error_deg = math.degrees(
                        direction.angle(annotation.direction_in_world)
                    )
                    if direction_error_deg > 90:
                        # hotfix
                        direction_error_deg = 180 - direction_error_deg

                    log.info(f"rotation about corridor: {rotation_about_corridor_deg} degrees")
                    log.info(f"startpoint_error_mm: {startpoint_error_mm:.02f} mm")
                    log.info(f"direction_error_deg: {direction_error_deg:.02f} degrees")

                    results["relative_rotation_deg"] = abs(math.degrees(relative_rotation))
                    results["rotation_about_corridor_deg"] = rotation_about_corridor_deg
                    results["corridor_startpoint_error_mm"] = startpoint_error_mm
                    results["corridor_angle_error_deg"] = direction_error_deg
                    results[
                        "corridor_circumscribing_radius_mm"
                    ] = geo_utils.radius_of_circumscribing_cylinder(
                        annotation.startpoint_in_world,
                        annotation.startpoint_in_world.lerp(annotation.endpoint_in_world, 0.8),
                        corridor,
                    )

                    corridor_annotation = LineAnnotation(
                        startpoint, startpoint + 200 * direction, volume=ct
                    )
                    corridor_annotation.save(
                        annotation_dir
                        / f"rot-{int(rotation_about_corridor_deg):03d}_{idx_1:03d}_{idx_2:03d}.mrk.json"
                    )

                    csv_writer.writerow(results)

                    # startpoint_error_mm < 3 and direction_error_deg < 2:
                    if rotation_about_corridor_deg >= 30:
                        _, _, _, _, corridor_mask_1 = model.analyze_image(
                            image_1, left_side, image_size
                        )
                        _, _, _, _, corridor_mask_2 = model.analyze_image(
                            image_2, left_side, image_size
                        )
                        corridor_image_1 = utils.combine_heatmap(image_1, corridor_mask_1)
                        corridor_image_2 = utils.combine_heatmap(image_2, corridor_mask_2)
                        image_path_1 = image_utils.save(
                            f"corridors/{specimen_name}_{proj_key_1}_corridor.png", corridor_image_1
                        )
                        image_path_2 = image_utils.save(
                            f"corridors/{specimen_name}_{proj_key_2}_corridor.png", corridor_image_2
                        )
                        frustum_1, image_mesh_1 = get_frustum_mesh(
                            index_from_world_1, pixel_size, image_path=image_path_1
                        )
                        frustum_2, image_mesh_2 = get_frustum_mesh(
                            index_from_world_2, pixel_size, image_path=image_path_2
                        )
                        corridor_length = 115
                        endpoint = startpoint + corridor_length * direction
                        corridor_mesh = pv.Cylinder(
                            startpoint.lerp(endpoint, 0.5),
                            direction,
                            radius=5,
                            height=corridor_length,
                        )
                        s1 = sources[idx_1]
                        s2 = sources[idx_2]
                        pl1 = get_detector_plane(index_from_world_1, pixel_size)
                        pl2 = get_detector_plane(index_from_world_2, pixel_size)
                        source_1 = pv.Sphere(center=s1, radius=10)
                        source_1 += pv.Line(s1, s1.join(startpoint).meet(pl1))
                        source_1 += pv.Line(s1, s1.join(endpoint).meet(pl1))
                        source_2 = pv.Sphere(center=s2, radius=10)
                        source_2 += pv.Line(s2, s2.join(startpoint).meet(pl2))
                        source_2 += pv.Line(s2, s2.join(endpoint).meet(pl2))

                        # plotter = pv.Plotter(off_screen=True, window_size=(1536, 2048))
                        plotter = pv.Plotter(off_screen=True, window_size=(1024, 1024))
                        plotter.set_background("white")
                        plotter.add_mesh(surface, color="white", opacity=1)
                        # plotter.add_mesh(corridor_annotation.get_mesh_in_world(), color="red")
                        plotter.add_mesh(corridor_mesh, color="red")
                        # plotter.add_mesh(frustum_1, color="blue")
                        # plotter.add_mesh(frustum_2, color="blue")
                        plotter.add_mesh(source_1, color="blue")
                        plotter.add_mesh(source_2, color="blue")
                        plotter.add_mesh(image_mesh_1, rgb=True, opacity=0.8)
                        plotter.add_mesh(image_mesh_2, rgb=True, opacity=0.8)
                        # plotter.add_axes()

                        cam_offset = geo.v(0, -900, -150)
                        cam_pos = annotation.startpoint_in_world + cam_offset
                        cam_focus = annotation.startpoint_in_world
                        cam_viewup = [0, -1, 0]
                        plotter.set_position(cam_pos)
                        plotter.set_focus(cam_focus)
                        plotter.set_viewup(cam_viewup)

                        animate = True
                        if animate:
                            nframe = 160
                            animations_dir = Path("animations")
                            animations_dir.mkdir(exist_ok=True)
                            animation_path = (
                                animations_dir / f"{specimen_name}_{proj_key_1}_{proj_key_2}.gif"
                            )
                            plotter.open_gif(str(animation_path), loop=0, fps=30)

                            rot_axis = np.array([0, -1, 0.1])
                            rot_axis = rot_axis / np.linalg.norm(rot_axis)

                            for theta in (
                                np.pi / 3 * np.sin(np.linspace(0, 2 * np.pi, nframe, endpoint=True))
                            ):
                                # Rotate about the y-axis by theta
                                plotter.camera_position = list(
                                    annotation.startpoint_in_world
                                    + geo.f(geo.Rotation.from_rotvec(theta * rot_axis)) @ cam_offset
                                )

                                plotter.set_focus(cam_focus)
                                # plotter.set_viewup(cam_viewup)
                                plotter.write_frame()
                            log.info(f"Saved animation to {animation_path}")
                            log.info(f"Exiting after first case while making figures.")
                            exit()
                        else:
                            screenshot = plotter.screenshot(return_img=True)
                            if screenshot is not None:
                                image_utils.save(
                                    f"screenshots/{specimen_name}_{proj_key_1}_{proj_key_2}.png",
                                    screenshot,
                                )
                            else:
                                log.debug("No screenshot")
                        plotter.close()

    f.close()
    results_file.close()
