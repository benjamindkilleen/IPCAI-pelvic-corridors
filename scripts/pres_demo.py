import deepdrr
from deepdrr.device import SimpleDevice
from deepdrr.annotations import LineAnnotation, FiducialList
from deepdrr import geo
import numpy as np
import click
from rich.logging import RichHandler
import imageio.v3 as iio
import imageio
import logging
from pathlib import Path
import math
from enum import Enum, auto
from deepdrr.utils import data_utils, image_utils
from rich.progress import track
import shutil
from pelphix.tools.screws import Screw_T16_L90 as Screw  # good luck finding
import pyvista as pv
import time


log = logging.getLogger("main")


class Acquisition(Enum):
    ap = auto()
    lateral = auto()
    inlet = auto()
    outlet = auto()
    oblique_left = auto()
    oblique_right = auto()


def get_APP(pelvis_keypoints: dict[str, geo.Point3D]) -> geo.FrameTransform:
    """Get the anterior pelvic plane coordinate (APP) frame.

    See https://www.nature.com/articles/s41598-019-49573-4 for details.

    Args:
        pelvis_keypoints: The pelvis keypoints in anatomical coordinates.
    """

    r_sps = pelvis_keypoints["r_sps"]
    l_sps = pelvis_keypoints["l_sps"]
    r_asis = pelvis_keypoints["r_asis"]
    l_asis = pelvis_keypoints["l_asis"]

    sps_midpoint = l_sps.lerp(r_sps, 0.5)
    asis_midpoint = l_asis.lerp(r_asis, 0.5)

    z_axis = (asis_midpoint - sps_midpoint).hat()
    x_axis_approx = (r_asis - l_asis).hat()

    y_axis = z_axis.cross(x_axis_approx).hat()
    x_axis = y_axis.cross(z_axis).hat()

    rot = np.stack([x_axis, y_axis, z_axis], axis=0)
    return geo.F.from_rt(rot.T, sps_midpoint)


inlet_angle = math.radians(45)  # about X
outlet_angle = math.radians(-40)  # about X
oblique_left_angle = math.radians(45)  # about Z
oblique_right_angle = math.radians(-45)  # about Z

canonical_views_in_APP = {
    Acquisition.ap: geo.v(0, 1, 0),
    Acquisition.lateral: geo.v(-1, 0, 0),
    Acquisition.inlet: geo.v(0, math.cos(inlet_angle), math.sin(inlet_angle)),
    Acquisition.outlet: geo.v(0, math.cos(outlet_angle), math.sin(outlet_angle)),
    Acquisition.oblique_left: geo.v(math.sin(oblique_left_angle), math.cos(oblique_left_angle), 0),
    Acquisition.oblique_right: geo.v(
        math.sin(oblique_right_angle), math.cos(oblique_right_angle), 0
    ),
}


def get_view_direction(
    view: Acquisition,
    world_from_APP: geo.FrameTransform,
    corridors: dict[str, LineAnnotation],
) -> geo.Vector3D:
    """Get a viewing direction in world coordinates.

    Args:
        view: The view to get.
        APP: The APP frame.
    """
    if view == Acquisition.ap:
        return world_from_APP @ geo.v(0, 1, 0)
    elif view == Acquisition.lateral:
        return world_from_APP @ geo.v(-1, 0, 0)
    elif view == Acquisition.inlet:
        return world_from_APP @ geo.v(0, math.sin(inlet_angle), math.cos(inlet_angle))
    elif view == Acquisition.outlet:
        return world_from_APP @ geo.v(0, math.sin(outlet_angle), math.cos(outlet_angle))
    elif view == Acquisition.oblique_left:
        return world_from_APP @ geo.v(
            math.cos(oblique_left_angle),
            math.sin(oblique_left_angle),
            0,
        )
    elif view == Acquisition.oblique_right:
        return world_from_APP @ geo.v(
            math.cos(oblique_right_angle), math.sin(oblique_right_angle), 0
        )
    elif view in corridors:
        return corridors[view].get_direction()
    else:
        raise ValueError(f"Unknown view: '{view}'")


@click.command()
@click.option(
    "--nmdid-dir",
    default="/home/killeen/datasets/OneDrive/sambhav/NMDID-ARCADE",
    help="Path to NMDID-ARCADE dataset",
)
@click.option(
    "--output-dir",
    default="images/pres_demo",
    help="Path to output directory",
)
def main(nmdid_dir: Path, output_dir: Path):
    nmdid_dir = Path(nmdid_dir).expanduser()
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    ct_path = (
        nmdid_dir
        / "nifti/THIN_BONE_TORSO/case-100114/THIN_BONE_TORSO_STANDARD_TORSO_Thorax_65008_11.nii.gz"
    )
    mesh_dir = nmdid_dir / "TotalSegmentator_mesh/THIN_BONE_TORSO/case-100114"
    annotation_dir = nmdid_dir / "2023-03-12_pelvis-annotations_ssm/case-100114"
    device = SimpleDevice(
        sensor_height=400, sensor_width=400, pixel_size=0.5, source_to_detector_distance=1000
    )

    ct = deepdrr.Volume.from_nifti(ct_path)
    log.debug(f"CT: {ct.shape}")
    ct.orient_patient()
    wire = deepdrr.vol.KWire.from_example()
    screw = Screw(density=0.06)

    landmark_points, names = data_utils.load_fcsv(annotation_dir / "landmarks.fcsv")
    landmarks: dict[str, geo.Point3D] = dict(
        (name, ct.world_from_anatomical @ geo.RAS_from_LPS @ geo.point(point))
        for name, point in zip(names, landmark_points)
    )
    corridors = dict(
        (
            p.stem,
            LineAnnotation.from_fcsv(
                p,
                anatomical_coordinate_system="RAS",
                world_from_anatomical=ct.world_from_anatomical,
            ),
        )
        for p in annotation_dir.glob("*.fcsv")
        if p.stem != "landmarks"
    )
    corridor = corridors["ramus_right"]
    corridor.endpoint = geo.RAS_from_LPS @ corridor.endpoint
    corridor.startpoint = geo.RAS_from_LPS @ corridor.startpoint

    bone_mesh = (
        pv.read(str(mesh_dir / "hip_left.stl"))
        + pv.read(str(mesh_dir / "hip_right.stl"))
        + pv.read(str(mesh_dir / "sacrum.stl"))
        + pv.read(str(mesh_dir / "femur_left.stl"))
        + pv.read(str(mesh_dir / "femur_right.stl"))
        + pv.read(str(mesh_dir / "vertebrae_L5.stl"))
    )
    bone_mesh.transform(geo.get_data(ct.world_from_anatomical @ geo.RAS_from_LPS), inplace=True)

    world_from_APP = get_APP(landmarks)
    view_direction = get_view_direction(Acquisition.oblique_left, world_from_APP, corridors)
    view_point = corridor.midpoint_in_world
    projector = deepdrr.Projector(
        [ct, wire], device=device, neglog=True, attenuate_outside_volume=True
    )
    projector.initialize()

    up_vector = ct.world_from_anatomical @ geo.v(0, 0, 1)
    device.set_view(
        view_point,
        view_direction,
        up=up_vector,
        source_to_point_fraction=0.9,
    )

    insertion = screw.length_mm() - 30
    wire.align(corridor.startpoint_in_world, corridor.endpoint_in_world, distance=insertion)
    screw.align(corridor.startpoint_in_world, corridor.endpoint_in_world, distance=insertion)
    perp_axis = screw.trajectory_in_world.get_direction().cross(view_direction).hat()

    plotter = pv.Plotter()
    wire.align(corridor.startpoint_in_world, corridor.endpoint_in_world, progress=0.9)
    plotter.add_mesh(bone_mesh, color="DEC765", opacity=0.5)
    plotter.add_mesh(screw.get_mesh_in_world(full=True), color="gray", opacity=1)
    plotter.add_mesh(wire.get_mesh_in_world(full=True, use_cached=False), color="gray", opacity=1)
    # plotter.add_mesh(device.get_mesh_in_world(), color="blue", opacity=1)
    plotter.set_background("white")
    plotter.show()
    exit()

    def run(basename: str):
        log.info(f"Rendering {basename}")
        device.set_view(view_point, view_direction, up=up_vector, source_to_point_fraction=0.9)
        image_utils.save(output_dir / f"{basename}.png", np.fliplr(projector()))
        # plotter = pv.Plotter()
        # plotter.add_mesh(bone_mesh, color="DEC765", opacity=0.5)
        # plotter.add_mesh(
        #     wire.get_mesh_in_world(full=True, use_cached=False), color="gray", opacity=1
        # )
        # plotter.add_mesh(device.get_mesh_in_world(), color="blue", opacity=1)
        # plotter.set_background("white")
        # plotter.show()

        cur_view_direction = view_direction.rotate(perp_axis, np.radians(30))
        frames = []
        thetas = np.radians(np.arange(0, 360, 1))
        for theta in track(thetas, description="Rotating"):
            device.set_view(
                view_point,
                cur_view_direction.rotate(view_direction, theta),
                up=up_vector,
                source_to_point_fraction=0.9,
            )
            frame = projector()
            frame = np.fliplr(frame)
            frame = image_utils.ensure_cdim(frame, 3)
            frame = image_utils.as_uint8(frame)
            frames.append(frame)

        vid_path = output_dir / f"{basename}.mp4"
        iio.imwrite(vid_path, frames, fps=30)

    # run("screw_correct")
    run("wire_correct")
    exit()

    screw.align(corridor.startpoint_in_world, corridor.endpoint_in_world, distance=insertion)
    wire.align(corridor.startpoint_in_world, corridor.endpoint_in_world, distance=insertion)
    screw.rotate(np.radians(-20) * perp_axis, center=screw.base_in_world)
    wire.rotate(np.radians(-20) * perp_axis, center=screw.base_in_world)
    # time.sleep(10)
    # run("screw_rotated_oop_-20")
    run("wire_rotated_oop_-20")

    screw.align(corridor.startpoint_in_world, corridor.endpoint_in_world, distance=insertion)
    wire.align(corridor.startpoint_in_world, corridor.endpoint_in_world, distance=insertion)
    screw.rotate(np.radians(20) * perp_axis, center=screw.base_in_world)
    wire.rotate(np.radians(20) * perp_axis, center=screw.base_in_world)
    # time.sleep(10)
    # run("screw_rotated_oop_+20")
    run("wire_rotated_oop_+20")

    projector.free()

    gif_path = output_dir / "source_to_detector_animation.gif"
    log.info(f"Saving animation to {gif_path}...")
    # iio.imwrite(gif_path, frames, duration=1, loop=0)


if __name__ == "__main__":
    main()
