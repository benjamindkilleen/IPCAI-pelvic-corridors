from __future__ import annotations
from datetime import datetime
import logging
import math
from typing import Any, Dict, Literal, Optional, Tuple, overload
import cv2
from deepdrr.utils import dicom_utils
import scipy
import subprocess as sp
from shutil import rmtree
from glob import iglob

from klampt.math import se3
from klampt import WorldModel
from klampt import IKSolver
from klampt.model import ik
from klampt import vis

import numpy as np
from deepdrr.device import Device
from deepdrr import geo
from deepdrr import Volume
from pathlib import Path
import pydicom
import pyvista as pv


log = logging.getLogger(__name__)


DEGREE_SIGN = "\N{DEGREE SIGN}"
DEBUG = False


# TODO:
# - Figure out how images are taken when source/detector not 180 apart and plan for it.


def clamp(num: float, min_value: float, max_value: float) -> float:
    return max(min(num, max_value), min_value)


class LoopX(Device):
    """A class for representing the Loop-X device from BrainLab.

    This class is for planning and simulating the Loop-X, given the parameters that determine its
    pose. It *does not* precisely model the device for existing acquisitions (see
    :class:`Acquisition`), but can be used to plan for future acquisitions and produce reasonable
    images for training deep neural networks.

    For more info, see: https://www.brainlab.com/surgery-products/overview-platform-products/robotic-intraoperative-mobile-cbct/
    For the user manual, email Benjamin Killeen at killeen@jhu.edu. Most of this info comes from that.
    - Panel size: 43.2 x 43.2 cm
    - Panel resolution: 2880 x 2880 pixels @ 150 micrometers/pixel (0.15mm/pix) (max, also 2x binning and 4x binning)
    - Source angle: [0, 683.3] degrees
    - Detector angle: [0, 688.5] degrees
    - Max rotational speed: 16 degrees/s, 7 degrees/s typical (can do up to 120)
    - Tilt is typically +/- 30 degrees, but can go +60 degrees for 2D source. Yaw rotation unlimited.
    - The Loop-X weights 520 kg, with a 203 kg gantry.
    - So if it gets to 0.5 m/s in 1s, a=0.5 m/s^2, and F = m * a = 520 kg * 0.5 m/s^2 = 1020 N
    As a design choice, we expose the same units that can be entered in the Loop-X control panel,
    but the underlying variables are in SI units (m, radians), to be compatible with the IK solver.
    NOTE: when exposing to the rest of deepdrr, units should be in (mm, radians).
    NOTE: robotworld is poorly named. This is really the frame at the Loop-X home position in meters.
    Attributes:

    Some quantities:
    - source to detector distance (focal length): 1247.36 +/- 0.46 mm (fully separated)
    - isocenter to detector plane 500.77 +/- 0.25 mm
    - source to isocenter: 747.09 +/- 0.63 mm

    # Remember to adjust for the vertices of the shutters


    FIXME: because of the way the robot is modeled, image coordinates are flipped in the x (width) direction.
    """

    markers = np.array(
        [
            [301.52, -501.57, 117.59],
            [158.36, -439.77, 110.92],
            [5.82, 594.93, -166.36],
            [-79.58, 495.47, -132.51],
            [-70.46, -230.06, 69.61],
            [-127.72, -137.72, 47.83],
            [-187.94, 218.71, -47.08],
        ]
    )

    # "loopx" is the frame of the Polaris markers. From the loop-x based calibration. Works well
    # enough for planning views and reproducing an approximate image, but should not be relied on to
    # precisely reproduce an acqusition, given the source angle params. For that, see the Acqusition
    # class.
    isocenter_from_loopx = geo.frame_transform(
        """
        0.02202494 -0.96339402 0.26718319 134.23276318
        0.05107703 0.26798333 0.96206864 -186.66779173
        -0.99845182 -0.00754258 0.05510962 504.90694616
        0.00000000 0.00000000 0.00000000 1.00000000
        """
    )
    loopx_from_isocenter = isocenter_from_loopx.inverse()

    # nominal focal length from calibration: 1369.185 mm

    BASE_PIXEL_SIZE_MM: float = 0.150
    BASE_SENSOR_HEIGHT: int = 2880
    BASE_SENSOR_WIDTH: int = 2880

    def __init__(
        self,
        data_dir: Optional[str] = None,
        urdf_file: str = "loop_x.urdf",
        world_from_device: Optional[geo.FrameTransform] = None,
        head_first: bool = True,
        supine: bool = True,
        source_angle: float = 180.0,
        detector_angle: float = 360.0,
        lateral: float = 0.0,
        longitudinal: float = 0.0,
        traction_yaw: float = 0.0,
        gantry_tilt: float = 0.0,
        binning: Literal["none", "2x", "4x", "8x"] = "none",
        sensor_height: Optional[int] = None,
        sensor_width: Optional[int] = None,
        q: Optional[np.ndarray] = None,
        flip_x: bool = True,
    ):
        """Initialize the Loop-X device.

        Args:
            data_dir: The directory where the URDF (and other data, maybe cad models) is stored. If not specified, defined
                relative the directory of this file as "../data".
            world_from_device: The transform from the world frame to the device frame. Note this is
                still in mm. LoopX maintains the conversion internally, so the rest of deepdrr can use
                this as expected. This defines the home position of the device. See `set_home()`.
            head_first: Whether the patient (or phantom) is lying head first. Ensure the device is set the same way. Defaults to True.
            supine: Whether the patient (or phantom) is lying supine (face up). Ensure the device is set the same way. Defaults to True.
            source_angle: The source angle of the device in degrees. Defaults to 180.0.
            detector_angle: The detector angle of the device in degrees. Defaults to 360.0.
            lateral: The lateral position of the device in cm. Defaults to 0.0.
            longitudinal: The longitudinal position of the device in cm. Defaults to 0.0.
            traction_yaw: The traction yaw of the device in degrees. Defaults to 0.0.
            gantry_tilt: The gantry tilt of the device in degrees. Defaults to 0.0.
            binning: The binning of the device. Determines the pixel size. Also determines the sensor size
                if None provided. "none" corresponds to default Loop-X resolution: 2880x2880 at 0.150 mm/pixel.
                Other options are "2x" and "4x" binning. Defaults to "none".
            sensor_height: The height of the sensor in pixels. Overrides binning if provided. This can also be set at any point by the user. Defaults to None.
            sensor_width: The width of the sensor in pixels. Overrides binning if provided. This can also be set at any point by the user. Defaults to None.
            flip_x: Whether to flip the x axis of the image. If False, the image will be flipped in the x direction. Defaults to True (matching real world).
                Only reason to set this to False is when using images generated with the old model.
        """
        if data_dir is not None:
            self.data_dir = Path(data_dir).expanduser().resolve()
        else:
            self.data_dir = (Path(__file__).parent.parent / "data").resolve()
        self.urdf_path = self.data_dir / urdf_file
        if not self.urdf_path.exists():
            raise FileNotFoundError(
                f"URDF file {self.urdf_path} not found. TODO: download automatically."
            )

        self.world_from_device = (
            geo.FrameTransform.identity(3)
            if world_from_device is None
            else geo.frame_transform(world_from_device)
        )

        # Used to determine the patient orientation.
        self.head_first = head_first
        self.supine = supine

        self.flip_x = flip_x

        # Robot model is mainly used during align, but joint limits are read from the URDF.
        self.world = WorldModel(str(self.urdf_path))
        self.robot = self.world.robot(0)
        # self.robot.saveFile(str(self.urdf_path.with_suffix(".rob")))

        self.lidx = dict((self.robot.link(i).name, i) for i in range(self.robot.numLinks()))
        log.debug(f"links: {self.lidx}")

        qmin, qmax = self.robot.getJointLimits()
        self.min_source_angle_rad = qmin[self.lidx["source_angle"]]
        self.max_source_angle_rad = qmax[self.lidx["source_angle"]]
        self.min_detector_angle_rad = qmin[self.lidx["detector_angle"]]
        self.max_detector_angle_rad = qmax[self.lidx["detector_angle"]]
        self.min_lateral_m = qmin[self.lidx["lateral"]]
        self.max_lateral_m = qmax[self.lidx["lateral"]]
        self.min_longitudinal_m = qmin[self.lidx["longitudinal"]]
        self.max_longitudinal_m = qmax[self.lidx["longitudinal"]]
        self.min_traction_yaw_rad = qmin[self.lidx["traction_yaw"]]
        self.max_traction_yaw_rad = qmax[self.lidx["traction_yaw"]]
        self.min_gantry_tilt_rad = qmin[self.lidx["gantry_tilt"]]
        self.max_gantry_tilt_rad = qmax[self.lidx["gantry_tilt"]]
        self.min_source_slider_m = qmin[self.lidx["source_slider"]]
        self.max_source_slider_m = qmax[self.lidx["source_slider"]]
        self.min_detector_slider_m = qmin[self.lidx["detector_slider"]]
        self.max_detector_slider_m = qmax[self.lidx["detector_slider"]]

        # Main parameters
        self.source_angle = source_angle
        self.detector_angle = detector_angle
        self.lateral = lateral
        self.longitudinal = longitudinal
        self.traction_yaw = traction_yaw
        self.gantry_tilt = gantry_tilt

        # Starting slider positions, in m.
        self.source_slider_m = 0.700
        self.detector_slider_m = -0.416

        # TODO: the home config is actually a constant and should be a class variable. Set it as such.
        self.home_config = self.robot.getConfig()
        if q is not None:
            self.robot.setConfig(q)

        # Set the binning.
        self.binning = binning
        binning_factor = {"none": 1, "2x": 2, "4x": 4, "8x": 8}[binning]
        self.pixel_size = self.BASE_PIXEL_SIZE_MM * binning_factor
        if sensor_height is None:
            self.sensor_height = self.BASE_SENSOR_HEIGHT // binning_factor
        else:
            self.sensor_height = sensor_height
        if sensor_width is None:
            self.sensor_width = self.BASE_SENSOR_WIDTH // binning_factor
        else:
            self.sensor_width = sensor_width

    @property
    def RAS_from_device(self) -> geo.FrameTransform:
        """The transform from the device frame to the RAS frame.
        The RAS frame is defined in terms of the device frame.
        """
        if self.head_first and self.supine:
            # R <- x, A <- z, S <- -y
            return geo.FrameTransform.from_rotation(np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]))
        elif self.head_first and not self.supine:
            # R <- -x, A <- -z, S <- -y
            return geo.FrameTransform.from_rotation(np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]]))
        elif not self.head_first and self.supine:
            # R <- -x, A <- z, S <- y
            return geo.FrameTransform.from_rotation(np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]]))
        elif not self.head_first and not self.supine:
            # R <- x, A <- -z, S <- y
            return geo.FrameTransform.from_rotation(np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]))
        else:
            raise ValueError("Invalid patient orientation.")

    @property
    def source_angle_rad(self) -> float:
        return self.robot.getConfig()[self.lidx["source_angle"]]

    @source_angle_rad.setter
    def source_angle_rad(self, value: float):
        if value < self.min_source_angle_rad or value > self.max_source_angle_rad:
            log.warning(
                f"Clamping source angle value of {value:.05f} to [{self.min_source_angle_rad}, {self.max_source_angle_rad}] rad."
            )
            value = clamp(value, self.min_source_angle_rad, self.max_source_angle_rad)
        q = self.robot.getConfig()
        q[self.lidx["source_angle"]] = value
        self.robot.setConfig(q)

    @property
    def detector_angle_rad(self) -> float:
        return self.robot.getConfig()[self.lidx["detector_angle"]]

    @detector_angle_rad.setter
    def detector_angle_rad(self, value: float):
        if value < self.min_detector_angle_rad or value > self.max_detector_angle_rad:
            log.warning(
                f"Clamping detector angle value of {value:.05f} to [{self.min_detector_angle_rad}, {self.max_detector_angle_rad}] rad."
            )
            value = clamp(value, self.min_detector_angle_rad, self.max_detector_angle_rad)
        q = self.robot.getConfig()
        q[self.lidx["detector_angle"]] = value
        self.robot.setConfig(q)

    @property
    def lateral_m(self) -> float:
        return self.robot.getConfig()[self.lidx["lateral"]]

    @lateral_m.setter
    def lateral_m(self, value: float):
        if value < self.min_lateral_m or value > self.max_lateral_m:
            log.warning(
                f"Clamping lateral value of {value:.05f} to [{self.min_lateral_m}, {self.max_lateral_m}] m."
            )
            value = clamp(value, self.min_lateral_m, self.max_lateral_m)
        q = self.robot.getConfig()
        q[self.lidx["lateral"]] = value
        self.robot.setConfig(q)

    @property
    def longitudinal_m(self) -> float:
        return self.robot.getConfig()[self.lidx["longitudinal"]]

    @longitudinal_m.setter
    def longitudinal_m(self, value: float):
        if value < self.min_longitudinal_m or value > self.max_longitudinal_m:
            log.warning(
                f"Clamping longitudinal value of {value:.05f} to [{self.min_longitudinal_m}, {self.max_longitudinal_m}] m."
            )
            value = clamp(value, self.min_longitudinal_m, self.max_longitudinal_m)
        q = self.robot.getConfig()
        q[self.lidx["longitudinal"]] = value
        self.robot.setConfig(q)

    @property
    def traction_yaw_rad(self) -> float:
        return self.robot.getConfig()[self.lidx["traction_yaw"]]

    @traction_yaw_rad.setter
    def traction_yaw_rad(self, value: float):
        if value < self.min_traction_yaw_rad or value > self.max_traction_yaw_rad:
            log.warning(
                f"Clamping traction yaw value of {value:.05f} to [{self.min_traction_yaw_rad}, {self.max_traction_yaw_rad}] rad."
            )
            value = clamp(value, self.min_traction_yaw_rad, self.max_traction_yaw_rad)
        q = self.robot.getConfig()
        q[self.lidx["traction_yaw"]] = value
        self.robot.setConfig(q)

    @property
    def gantry_tilt_rad(self) -> float:
        return self.robot.getConfig()[self.lidx["gantry_tilt"]]

    @gantry_tilt_rad.setter
    def gantry_tilt_rad(self, value: float):
        if value < self.min_gantry_tilt_rad or value > self.max_gantry_tilt_rad:
            log.warning(
                f"Clamping gantry tilt value of {value:.05f} to [{self.min_gantry_tilt_rad}, {self.max_gantry_tilt_rad}] rad."
            )
            value = clamp(value, self.min_gantry_tilt_rad, self.max_gantry_tilt_rad)
        q = self.robot.getConfig()
        q[self.lidx["gantry_tilt"]] = value
        self.robot.setConfig(q)

    @property
    def source_slider_m(self) -> float:
        return self.robot.getConfig()[self.lidx["source_slider"]]

    @source_slider_m.setter
    def source_slider_m(self, value: float):
        if value < self.min_source_slider_m or value > self.max_source_slider_m:
            log.warning(
                f"Clamping source slider value of {value:.05f} to [{self.min_source_slider_m}, {self.max_source_slider_m}] m."
            )
            value = clamp(value, self.min_source_slider_m, self.max_source_slider_m)
        q = self.robot.getConfig()
        q[self.lidx["source_slider"]] = value
        self.robot.setConfig(q)

    @property
    def detector_slider_m(self) -> float:
        return self.robot.getConfig()[self.lidx["detector_slider"]]

    @detector_slider_m.setter
    def detector_slider_m(self, value: float):
        if value < self.min_detector_slider_m or value > self.max_detector_slider_m:
            log.warning(
                f"Clamping detector slider value of {value:.05f} to [{self.min_detector_slider_m}, {self.max_detector_slider_m}] m."
            )
            value = clamp(value, self.min_detector_slider_m, self.max_detector_slider_m)
        q = self.robot.getConfig()
        q[self.lidx["detector_slider"]] = value
        self.robot.setConfig(q)

    @property
    def source_angle(self) -> float:
        """Angular position of the source in degrees."""
        return math.degrees(self.source_angle_rad)

    @source_angle.setter
    def source_angle(self, value: float):
        """Set the source angle of the device in degrees."""
        self.source_angle_rad = math.radians(value)

    @property
    def detector_angle(self) -> float:
        """Angular position of the detector in degrees."""
        return math.degrees(self.detector_angle_rad)

    @detector_angle.setter
    def detector_angle(self, value: float):
        """Set the detector angle of the device in degrees."""
        self.detector_angle_rad = math.radians(value)

    @property
    def lateral(self) -> float:
        """The lateral position of the device in cm."""
        return self.lateral_m * 100.0

    @lateral.setter
    def lateral(self, value: float):
        """Set the lateral position of the device in cm."""
        self.lateral_m = value / 100.0

    @property
    def longitudinal(self) -> float:
        """The longitudinal position of the device in cm."""
        return self.longitudinal_m * 100.0

    @longitudinal.setter
    def longitudinal(self, value: float):
        """Set the longitudinal position of the device in cm."""
        self.longitudinal_m = value / 100.0

    @property
    def traction_yaw(self) -> float:
        """The traction yaw of the device in degrees."""
        return math.degrees(self.traction_yaw_rad)

    @traction_yaw.setter
    def traction_yaw(self, value: float):
        """Set the traction yaw of the device in degrees."""
        self.traction_yaw_rad = math.radians(value)

    @property
    def gantry_tilt(self) -> float:
        """The gantry tilt of the device in degrees."""
        return math.degrees(self.gantry_tilt_rad)

    @gantry_tilt.setter
    def gantry_tilt(self, value: float):
        """Set the gantry tilt of the device in degrees."""
        self.gantry_tilt_rad = math.radians(value)

    def set_pose(
        self,
        source_angle: Optional[float] = None,
        detector_angle: Optional[float] = None,
        lateral: Optional[float] = None,
        longitudinal: Optional[float] = None,
        traction_yaw: Optional[float] = None,
        gantry_tilt: Optional[float] = None,
    ) -> None:
        """Move the device to the given pose, as read directly off the device.
        This is mostly a redundant function, because we use setters to do all the clamping and converting.
        This pose can be read/set by clicking on the loop icon in the top right of the screen.
        Args:
            source_angle: The source angle of the device in degrees.
            detector_angle: The detector angle of the device in degrees.
            lateral: The lateral position of the device in cm.
            longitudinal: The longitudinal position of the device in cm.
            traction_yaw: The traction yaw of the device in degrees.
            gantry_tilt: The gantry tilt of the device in degrees.
        """
        if source_angle is not None:
            self.source_angle = source_angle
        if detector_angle is not None:
            self.detector_angle = detector_angle
        if lateral is not None:
            self.lateral = lateral
        if longitudinal is not None:
            self.longitudinal = longitudinal
        if traction_yaw is not None:
            self.traction_yaw = traction_yaw
        if gantry_tilt is not None:
            self.gantry_tilt = gantry_tilt

    def get_pose(self) -> Dict[str, float]:
        """Get the pose in cm/degrees for the device."""
        return dict(
            source_angle=self.source_angle,
            detector_angle=self.detector_angle,
            lateral=self.lateral,
            longitudinal=self.longitudinal,
            traction_yaw=self.traction_yaw,
            gantry_tilt=self.gantry_tilt,
        )

    def get_config(self) -> Dict[str, Any]:
        """Get a dictionary of keyword arguments to recreate the current state.
        ```
        LoopX(**loop.get_config())
        ```
        Not necessarily json-friendly, so use :func:`jsonable`.
        """
        return dict(
            data_dir=self.data_dir,
            world_from_device=self.world_from_device,
            head_first=self.head_first,
            supine=self.supine,
            source_angle=self.source_angle,
            detector_angle=self.detector_angle,
            lateral=self.lateral,
            longitudinal=self.longitudinal,
            traction_yaw=self.traction_yaw,
            gantry_tilt=self.gantry_tilt,
            binning=self.binning,
            q=self.robot.getConfig(),
        )

    def set_config(self, config: Dict[str, Any] = {}, **kwargs) -> None:
        """Set the device to the pose defined in config, as returned by get_config()."""
        config.update(kwargs)

        if "world_from_device" in config:
            self.world_from_device = geo.frame_transform(config["world_from_device"])
        if "head_first" in config:
            self.head_first = config["head_first"]
        if "supine" in config:
            self.supine = config["supine"]

        if "q" in config:
            q = list(map(float, config["q"]))
            if len(q) == len(self.lidx) - 1:
                # Hot fix because robot model was missing a joint during
                q.insert(self.lidx["isocenter"], 0.0)

            self.robot.setConfig(q)
        else:
            self.set_pose(
                source_angle=config.get("source_angle"),
                detector_angle=config.get("detector_angle"),
                lateral=config.get("lateral"),
                longitudinal=config.get("longitudinal"),
                traction_yaw=config.get("traction_yaw"),
                gantry_tilt=config.get("gantry_tilt"),
            )

        if "binning" in config:
            assert self.binning == config["binning"], "if this not true, need to create new device"

    def get_pose_si(self) -> Dict[str, float]:
        """Get the dictionary of pose values in SI units."""
        return dict(
            source_angle_rad=self.source_angle_rad,
            detector_angle_rad=self.detector_angle_rad,
            lateral_m=self.lateral_m,
            longitudinal_m=self.longitudinal_m,
            traction_yaw_rad=self.traction_yaw_rad,
            gantry_tilt_rad=self.gantry_tilt_rad,
        )

    @property
    def world_from_isocenter(self) -> geo.FrameTransform:
        """Get the transform from the world to the imaging center."""
        return self.world_from_device @ self._get_transform_mm("isocenter")

    @property
    def world_from_loopx(self) -> geo.FrameTransform:
        """Get the transform from the world to the Loop-X markers frame."""
        return self.world_from_isocenter @ self.isocenter_from_loopx

    @property
    def device_from_camera3d(self) -> geo.FrameTransform:
        """Get the camera3d pose (in mm)."""

        #  Get source/detector position. Rotate the source frame to point toward the detector.
        # log.warning("TODO: don't depend on source_ray. Invalid if manual pose is set.")
        return self._get_transform_mm("source_ray")

    @property
    def world_from_source(self) -> geo.FrameTransform:
        """Get the transform from the source to the world frame."""
        return self.world_from_device @ self._get_transform_mm("source")

    @property
    def source_center_in_world(self) -> geo.Point3D:
        """Get the source position in world coordinates."""
        return self.world_from_source.o

    @property
    def world_from_gantry(self) -> geo.FrameTransform:
        """Get the transform from the gantry to the world frame."""
        return self.world_from_device @ self._get_transform_mm("gantry_tilt")

    @property
    def source_slider_in_world(self) -> geo.Point3D:
        """Get the source slider position in world coordinates."""
        return self.world_from_device @ self._get_transform_mm("source_slider").o

    @property
    def detector_slider_in_world(self) -> geo.Point3D:
        """Get the detector slider position in world coordinates."""
        return self.world_from_device @ self._get_transform_mm("detector_slider").o

    @property
    def world_from_detector(self) -> geo.FrameTransform:
        """Get the transform from the detector to the world frame."""
        return self.world_from_device @ self._get_transform_mm("detector")

    @property
    def detector_center_in_world(self) -> geo.Point3D:
        """Get the detector position in world coordinates."""
        return self.world_from_detector.o

    @property
    def isocenter_in_device(self) -> geo.Point3D:
        """Get the source slider position in world coordinates."""
        return self._get_transform_mm("isocenter").o

    @property
    def isocenter_in_world(self) -> geo.Point3D:
        """Get the imaging center position in world coordinates."""
        return self.world_from_device @ self.isocenter_in_device

    def _get_source_ray_from_detector(self) -> geo.FrameTransform:
        robotworld_from_detector = self._get_transform_m("detector")
        robotworld_from_source_ray = self._get_transform_m("source_ray")
        return robotworld_from_source_ray.inv @ robotworld_from_detector

    def _get_transform_m(self, link: str) -> geo.FrameTransform:
        return geo.FrameTransform(se3.ndarray(self.robot.link(link).getTransform()))

    def _get_transform_mm(self, link: str) -> geo.FrameTransform:
        f = self._get_transform_m(link)
        f.t = f.t * 1000
        return f

    """To position the virtual loop-x, fcal is publishing the LoopXGantryToReference, or the
    Reference_from_LoopXGantry transform. LoopXGantry corresponds to the gantry link "gantry_tilt".
    And deepdrr's "world" is the reference frame. So we are given gantry_tilt 
    """

    def reposition(
        self,
        world_from_loopx: geo.FrameTransform,
        source_angle: float,
        detector_angle: float,
        lateral: float,
        longitudinal: float,
        traction_yaw: float,
        gantry_tilt: float,
    ) -> None:
        """Reposition the virtual loop-x to the given reference frame.
        The current Loop-X pose, except for source/detector angle is also required, since the given
        transform is to the gantry, not the linkmount.
        Args:
            world_from_loopx: The pose of the Loop-X markers frame in the DeepDRR world.
            **kwargs: Additional keyword arguments are passed to :func:`set_pose`. It is recommended to provide at
                least the lateral, longitudinal, traction_yaw, and gantry_tilt arguments, to specify
                the pose of the gantry.
        """
        self.source_angle = source_angle
        self.detector_angle = detector_angle
        self.lateral = lateral
        self.longitudinal = longitudinal
        self.traction_yaw = traction_yaw
        self.gantry_tilt = gantry_tilt

        # device_from_source_angle = self._get_transform("source_angle")
        device_from_isocenter = self._get_transform_mm("isocenter")
        self.world_from_device = (
            world_from_loopx @ self.loopx_from_isocenter @ device_from_isocenter.inverse()
        )

    def _get_aspect_ratio(
        self, source_ray_from_detector: Optional[geo.FrameTransform] = None
    ) -> float:
        """Get the aspect ratio of the current pose.
        Args:
            source_ray_from_detector (Optional[geo.FrameTransform]): Current cam from detector transform, if known.
        Returns:
            float: The aspect ratio of the current pose.
        """
        if source_ray_from_detector is None:
            source_ray_from_detector = self._get_source_ray_from_detector()
        theta = (source_ray_from_detector @ geo.v(0, 0, 1)).angle(geo.v(0, 0, 1))
        return math.cos(theta)

    @property
    def aspect_ratio(self) -> float:
        """The aspect ratio of the current pose."""
        return self._get_aspect_ratio()

    @property
    def camera_intrinsics(self) -> geo.CameraIntrinsicTransform:
        """Get the camera intrinsics for the current pose.
        Expects inputs to be in mm.
        """
        # The "detector" frame has the z-axis perpendicular to sensor, pointing outward (away from
        # source). The "source_ray" frame has z-axis pointing toward the detector origin, if the
        # virtual Loop-X has been positioned correctly (e.g. with self.align()).
        source_ray_from_detector_m = self._get_source_ray_from_detector()
        f_mm = np.linalg.norm(source_ray_from_detector_m[0:3, 3]) * 1000
        f = f_mm / self.pixel_size
        cx = self.sensor_width / 2
        cy = self.sensor_height / 2
        a = self._get_aspect_ratio(source_ray_from_detector_m)

        # TODO: aspect ratio appears to be unchanged. Figure out why.
        # K = np.array([[f, 0, cx], [0, a * f, cy], [0, 0, 1]])
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])

        if self.flip_x:
            # Correct for the fact that X-axis points to the left in Loop-X.
            flip_x = np.array(
                [
                    [-1, 0, self.sensor_width],
                    [0, 1, 0],
                    [0, 0, 1],
                ]
            )
            intrinsics = geo.CameraIntrinsicTransform(flip_x @ K)
        else:
            intrinsics = geo.CameraIntrinsicTransform(K)
        intrinsics.sensor_height = self.sensor_height
        intrinsics.sensor_width = self.sensor_width

        return intrinsics

    @property
    def principle_ray(self) -> geo.Vector3D:
        """Get the (unit-vector) principle ray for the current pose in device (mm) coordinates."""
        robotworld_from_source_ray = self._get_transform_m("source_ray")
        # Don't need to convert to mm because unit vector.
        return (robotworld_from_source_ray @ geo.v(0, 0, 1)).normalized()

    @property
    def source_to_detector_distance(self) -> float:
        """Get the current distance between the source and the detector in mm."""
        source_ray_from_detector_m = self._get_source_ray_from_detector()
        return np.linalg.norm(source_ray_from_detector_m.t) * 1000.0

    @property
    def robotworld_from_world(self) -> geo.FrameTransform:
        """The "device" frame is measured in millimeters, so convert to meters."""
        return geo.m_from_mm @ self.device_from_world

    @property
    def world_from_robotworld(self) -> geo.FrameTransform:
        """The "device" frame is measured in millimeters, so convert to meters."""
        return self.world_from_device @ geo.mm_from_m

    def align(
        self,
        point_in_world: geo.Point3D,
        direction_in_world: geo.Vector3D,
        traction_yaw: float = 0.0,
        traction_yaw_tolerance: float = 10.0,
        detector_to_point_distance: float = 550.0,
        detector_to_point_tolerance: float = 20.0,
        ray_angle: float = 0.0,
        ray_angle_tolerance: float = 30.0,
        min_gantry_tilt: Optional[float] = None,
        max_gantry_tilt: Optional[float] = None,
        hover: bool = False,
        tolerance: float = 1e-3,
        max_iters: int = 100,
    ) -> bool:
        """Position the device such that the principle ray aligns with the given ray.
        TODO: add magnification factor, which constrains the sliders.
        Args:
            point: A (non-unique) point on the desired principle ray, in world coordinates.
            direction: The direction of the desired principle ray (as pointing from source to detector), in world coordinates.
            traction_yaw: The desired traction yaw of the loop-x, in degrees.
            traction_yaw_tolerance: The tolerance for the traction yaw, in degrees.
            detector_to_point_distance: The desired distance between the detector and the point, in mm. Smaller values
                "zoom out" or make the anatomy appear smaller.
            detector_to_point_tolerance: The tolerance for the distance between the detector and the point, in mm.
            ray_angle: The desired angle of the principle ray relative to centered, in degrees.
            ray_angle_tolerance: The tolerance for the angle of the principle ray relative to
                centered, in degrees. Set to 0 for 180 degree separation between source and detector.
            min_gantry_tilt: The minimum gantry tilt, in degrees. If not provided, device defaults are used.
            max_gantry_tilt: The maximum gantry tilt, in degrees. If not provided, device defaults are used.
            hover: If True, the robot will be allowed to move freely in Z. This is unrealistic but convenient,
                and it is the same as moving the table.
            tolerance: The tolerance for the IK solver.
            max_iters (int): The maximum number of iterations for the IK solver.
        Returns:
            True if the device was successfully aligned, False otherwise.
        """

        # Copies the tm pose into the robot model.
        # Define the two-part objective for placing the source and the detector in (robot space)
        # TODO: remove robotworld_from_world, because of bug in m_from_mm
        q1 = self.robotworld_from_world @ geo.p(point_in_world)
        v = (self.robotworld_from_world @ geo.v(direction_in_world)).normalized()
        q2 = q1 + 0.1 * v

        q1 = list(q1)
        q2 = list(q2)
        p1 = [0, 0, 0]
        p2 = [0, 0, 0.1]

        source_slider = self.robot.link("source_slider")
        detector_slider = self.robot.link("detector_slider")
        source_obj = ik.objective(source_slider, local=[p1, p2], world=[q1, q2])
        detector_obj = ik.objective(detector_slider, local=[p1, p2], world=[q1, q2])

        # TODO(killeen): solve the kinematics with added constraints, minimizing:
        #   - traction yaw
        #   - lateral/longitudinal movement from initial position
        #   - (maximize) angular distance between source and detector
        # Can do this with a nonlinear solver, since we have the Jacobian for the joints. Just have to
        # differentiate the added terms.

        # Convert custom constraints to SI.
        detector_distance_m = detector_to_point_distance / 1000.0
        detector_tolerance_m = detector_to_point_tolerance / 1000.0
        traction_yaw_rad = math.radians(traction_yaw)
        traction_yaw_tolerance_rad = math.radians(traction_yaw_tolerance)
        ray_angle_rad = math.radians(ray_angle)
        ray_angle_tolerance_rad = math.radians(ray_angle_tolerance)

        # Constrain the detector slider to be at the given distance from the point.
        # But do not constrain the source slider, otherwise the view may not be achievable.
        # Recall -Z is toward the source.
        qmin, qmax = self.robot.getJointLimits()
        qmin[self.lidx["detector_slider"]] = -detector_distance_m - detector_tolerance_m
        qmax[self.lidx["detector_slider"]] = -detector_distance_m + detector_tolerance_m
        qmin[self.lidx["traction_yaw"]] = traction_yaw_rad - traction_yaw_tolerance_rad
        qmax[self.lidx["traction_yaw"]] = traction_yaw_rad + traction_yaw_tolerance_rad
        qmin[self.lidx["source_ray"]] = ray_angle_rad - ray_angle_tolerance_rad
        qmax[self.lidx["source_ray"]] = ray_angle_rad + ray_angle_tolerance_rad
        qmin[self.lidx["detector_ray"]] = ray_angle_rad - ray_angle_tolerance_rad
        qmax[self.lidx["detector_ray"]] = ray_angle_rad + ray_angle_tolerance_rad

        if min_gantry_tilt is not None:
            qmin[self.lidx["gantry_tilt"]] = math.radians(min_gantry_tilt)
        if max_gantry_tilt is not None:
            qmax[self.lidx["gantry_tilt"]] = math.radians(max_gantry_tilt)

        qmin[0] = 0.0
        qmin[1] = 0.0
        qmin[2] = 0.0
        qmin[3] = 0.0
        qmin[4] = 0.0
        qmin[5] = 0.0
        qmax[0] = 0.0
        qmax[1] = 0.0
        qmax[2] = 0.0
        qmax[3] = 0.0
        qmax[4] = 0.0
        qmax[5] = 0.0
        if hover:
            qmin[self.lidx["base2"]] = -np.inf
            qmax[self.lidx["base2"]] = np.inf
        # log.debug(f"qmax: {qmax}")
        # log.debug(f"q: {self.robot.getConfig()}")

        # TODO: if solver fails, increase the tolerance until it succeeds.

        # Solve the iK problem
        solver = IKSolver(self.robot)
        solver.add(source_obj)
        solver.add(detector_obj)
        solver.setTolerance(tolerance)  # can be bigger maybe?
        solver.setMaxIters(max_iters)
        solver.setJointLimits(qmin, qmax)

        # solver.setBiasConfig(self.robot.getConfig()) # breaks things
        res = solver.solve()
        min_max_achieved = np.stack([qmin, qmax, self.robot.getConfig()], axis=1)
        # log.debug(f"min, max, achieved: {np.array_str(min_max_achieved, precision=3)}")
        return res

    """
    Visualizing/reporting
    """

    def show(self):
        """Show the robot model in a separate window.
        See examples at https://github.com/krishauser/Klampt-examples/blob/master/Python3/demos/vis_template.py.
        """
        w = self.world
        vis.add("world", w)
        vis.run()

    def __repr__(self):
        return (
            f"LoopX(\n"
            f"    source_angle={self.source_angle},\n"
            f"    detector_angle={self.detector_angle},\n"
            f"    lateral={self.lateral},\n"
            f"    longitudinal={self.longitudinal},\n"
            f"    traction_yaw={self.traction_yaw},\n"
            f"    gantry_tilt={self.gantry_tilt},\n"
            f")"
        )

    def __str__(self):
        # TODO: subtract 2pi from angles to reasonable range.
        return (
            f"Loop-X:\n"
            f"    source_angle:   {self.source_angle: .02f}{DEGREE_SIGN}\n"
            f"    detector_angle: {self.detector_angle: .02f}{DEGREE_SIGN}\n"
            f"    lateral:        {self.lateral: .02f} cm\n"
            f"    longitudinal:   {self.longitudinal: .02f} cm\n"
            f"    traction_yaw:   {self.traction_yaw: .02f}{DEGREE_SIGN}\n"
            f"    gantry_tilt:    {self.gantry_tilt: .02f}{DEGREE_SIGN}\n"
        )


def _get_principle_ray(source_angle, detector_angle) -> geo.Vector3D:
    theta_s = np.radians(source_angle)
    theta_d = np.radians(detector_angle)
    s = geo.p(np.sin(theta_s), 0, np.cos(theta_s))
    d = geo.p(np.sin(theta_d), 0, np.cos(theta_d))
    r = d - s
    return r.hat()


class Acquisition(object):
    """A single slice from the Loop-X, with associated functions for recovering pose.

    This is intended to be used for 2D navigated acquisitions, but many of the tags apply to other
    types of acquisitions as well.

    All positions are with respect to the "ImagingRing" coordinate system.
    """

    # From optimization with backprojection based on acquisitions. Decent but still has tracker error.
    """
    0.05898765 -0.96376562 0.26014680 112.59315491
    0.08250834 0.26441586 0.96087283 -222.82095337
    -0.99484313 -0.03521538 0.09511592 498.04992676
    0.00000000 0.00000000 0.00000000 1.00000000
    """

    # From nonlinear pointer-based optimization (2020-10-30pointer calib 2)
    """
    0.03881794 -0.96301102 0.26665135 122.08823706
    0.07719078 0.26894498 0.96005738 -218.03350043
    -0.99626038 -0.01668443 0.08477546 497.64574930
    0.00000000 0.00000000 0.00000000 1.00000000
    """

    # From the CT export. Also suffers from tracker error.
    ring_from_loopx = geo.frame_transform(
        """
        0.03439477 -0.96211112 0.27047917 122.83479309
        0.07905842 0.27241051 0.95892757 -221.14457703
        -0.99627650 -0.01159846 0.08543249 495.09738159
        0.00000000 0.00000000 0.00000000 1.00000000
        """
    )

    loopx_from_ring = ring_from_loopx.inverse()

    # From DICOM export. I think this might depend on the patient orientation. This one is from head
    # first, supine.
    front_markers_in_scanner = np.array(
        [
            [-574.488, -162.097, 0.07],
            [-478.854, -38.705, -0.227],
            [-214.57, 129.217, -0.075],
            [-108.357, 160.672, -0.118],
            [263.81, 125.713, 0.07],
            [516.277, -52.702, 0.212],
            [596.181, -162.097, 0.07],
        ]
    )

    # Measured by polaris, but reordered to match the above.
    front_markers_in_loopx = np.array(
        [
            [301.52, -501.57, 117.59],
            [158.36, -439.77, 110.92],
            [-70.46, -230.06, 69.61],
            [-127.72, -137.72, 47.83],
            [-187.94, 218.71, -47.08],
            [-79.58, 495.47, -132.51],
            [5.82, 594.93, -166.36],
        ]
    )

    # From point correspondence of the above two.
    loopx_from_scanner = geo.f(
        """
        -0.25264031 -0.96490538 0.07162733 0.00000546
        0.93659043 -0.22530280 0.26839736 -0.00142892
        -0.24284025 0.13489348 0.96064168 0.00000182
        0.00000000 0.00000000 0.00000000 1.00000000
        """
    )
    scanner_from_loopx = loopx_from_scanner.inverse()

    # patient array being the frame of reference used by BrainLab
    # This was done by reading from CT, with a transform error of 0.29 +/- 0.04 mm,
    # and a projection error of 1.76 +/- 0.43 pixels.
    markers_in_patient_array = np.array(
        [
            [61.189, -37.823, 13.441],
            [-55.234, -49.173, 13.258],
            [64.216, 85.61, 13.63],
            [-73.923, 86.184, 13.568],
        ]
    )

    # "reference" is the patient array object, but measured from a different frame (custom ROM).
    # This registration was done from CT.
    patient_array_from_reference = geo.f(
        """0.21712711 0.97612822 0.00542996 -0.93804264
        -0.91629106 0.20189297 0.34590420 21.19936371
        0.33655068 -0.08008064 0.93825418 13.47446060
        0.00000000 0.00000000 0.00000000 1.00000000"""
    )
    reference_from_patient_array = patient_array_from_reference.inverse()

    # row, col pixel size at detector in mm
    pixel_size: Tuple[float, float]

    M_PP: Dict[str, geo.FrameTransform] = {
        "HFS": geo.f("1 0 0 0 0 0 1 0 0 -1 0 0 0 0 0 1"),
        "HFP": geo.f("-1 0 0 0 0 0 1 0 0 1 0 0 0 0 0 1"),
        "FFS": geo.f("-1 0 0 0 0 0 -1 0 0 -1 0 0 0 0 0 1"),
        "FFP": geo.f("1 0 0 0 0 0 -1 0 0 1 0 0 0 0 0 1"),
        "HFDL": geo.f("0 -1 0 0 0 0 1 0 -1 0 0 0 0 0 0 1"),
        "HFDR": geo.f("0 1 0 0 0 0 1 0 1 0 0 0 0 0 0 1"),
        "FFDL": geo.f("0 1 0 0 0 0 -1 0 -1 0 0 0 0 0 0 1"),
        "FFDR": geo.f("0 -1 0 0 0 0 -1 0 1 0 0 0 0 0 0 1"),
    }

    @property
    def patient_array_from_world(self) -> geo.FrameTransform:
        return self.world_from_patient_array.inverse()

    def __init__(
        self,
        path: Path,
        world_from_patient_array: Optional[geo.FrameTransform] = None,
    ):
        """Load the acquisition from a DICOM file.

        Meant to be for navigated acquisitions.

        Args:
            path: Path to the DICOM file.
            world_from_patient_array: Optional transform from the DICOM patient coordinate system to the world coordinate system.
        """
        self.path = Path(path).expanduser()
        self.world_from_patient_array = (
            geo.FrameTransform.identity(3)
            if world_from_patient_array is None
            else geo.f(world_from_patient_array)
        )

        ds = pydicom.dcmread(self.path)
        self.source_angle = float(ds[0x30BB2084].value)
        self.detector_angle = float(ds[0x30BB2086].value)
        self.gantry_tilt = float(ds[0x30BB2088].value)
        self.pixel_array = ds.pixel_array

        # Get the image
        image = self.pixel_array.astype(np.float32)
        self.image = image / (2**16 - 1)

        self.source_in_ring: geo.Point3D = geo.point(ds[0x30BB2080].value)
        self.detector_origin_in_ring: geo.Point3D = geo.point(ds[0x30BB2081].value)
        self.col_direction_in_ring: geo.Vector3D = geo.vector(ds[0x30BB2082].value)
        self.row_direction_in_ring: geo.Vector3D = geo.vector(ds[0x30BB2083].value)
        self.pixel_size = tuple(ds[0x00181164].value)  # (row,col)
        self.sensor_width = image.shape[1]
        self.sensor_height = image.shape[0]

        self.principle_ray = _get_principle_ray(self.source_angle, self.detector_angle)
        self.detector_normal = self.row_direction_in_ring.cross(self.col_direction_in_ring)
        self.detector_plane = geo.plane(self.detector_origin_in_ring, self.detector_normal)

        # Actual projection matrix from the dicom
        self.patient_position = ds[0x00185100].value

        # the patient array is the 4-sphere marker. It should be fixed with respect to the table.
        self.navigated = 0x30BB2050 in ds
        if self.navigated:
            self.calibration_matrix_front = np.array(ds[0x30BB2051].value).reshape(4, 4)
            self.marker_points_front = np.array(ds[0x30BB2052].value).reshape(-1, 3)
            self.calibration_matrix_back = np.array(ds[0x30BB2053].value).reshape(4, 4)
            self.marker_points_back = np.array(ds[0x30BB2054].value).reshape(-1, 3)
            self.patient_to_scanner_matrix = np.array(ds[0x30BB2055].value).reshape(4, 4)
        else:
            self.calibration_matrix_front = None
            self.marker_points_front = None
            self.calibration_matrix_back = None
            self.marker_points_back = None
            self.patient_to_scanner_matrix = None

        # "TTCS" has some axes negated compared to LoopX. This is M_IDT in the manual.
        # I don't think we care about this.
        # Pretty sure scanner is just the LPS coordinate system of any DICOM taken.
        tabletop_from_LPS = self.M_PP[self.patient_position]  # LPS
        ring_from_tabletop = geo.frame_transform(
            """
            -1 0 0 0
            0 -1 0 0
            0 0 1 0
            0 0 0 1
            """
        )
        self.ring_from_LPS = ring_from_tabletop @ tabletop_from_LPS
        self.LPS_from_ring = self.ring_from_LPS.inverse()

        H, W = image.shape

        # Decompose the projection matrix. This is the index_from_scanner matrix,
        # so we need to keep the patient orientation in mind.
        # From https://ksimek.github.io/2012/08/14/decompose/
        self.projection_matrix = np.array(ds[0x30BB2073].value).reshape(4, 4)
        P = self.projection_matrix[[0, 1, 3], :]  # Remove the Z row?
        M = P[:, 0:3]
        K, R = scipy.linalg.rq(M)
        T = np.diag(np.sign(np.diag(K)))
        if np.linalg.det(T) < 0:
            T[1, 1] *= -1
        K = K @ T
        R = T @ R

        # If image x-axis and camera x-axis point in opposite direcitons
        # K[:, 0] = -K[:, 0]
        # R[0, :] = -R[0, :]

        # If image y-axis and camera y-axis point in opposite direcitons
        # K[:, 1] = -K[:, 1]
        # R[1, :] = -R[1, :]

        if np.linalg.det(R) < 0:
            R = -R

        t = np.linalg.inv(K) @ P[:, 3]
        pixel_from_mm = geo.F.from_scaling(1 / self.pixel_size[0], dim=2)
        K = pixel_from_mm.data @ K
        K /= K[2, 2]

        self.intrinsic = geo.CameraIntrinsicTransform(
            K,
            sensor_height=self.sensor_height,
            sensor_width=self.sensor_width,
        )
        self.camera_from_LPS = geo.F.from_rt(R, t)
        self.index_from_LPS = geo.CameraProjection(self.intrinsic, self.camera_from_LPS)
        self.index_from_scanner = self.index_from_LPS @ self.LPS_from_scanner

    @property
    def LPS_from_scanner(self) -> geo.FrameTransform:
        if not self.navigated:
            raise ValueError("Not a navigated scan.")
        return geo.FrameTransform(self.calibration_matrix_front)

    @property
    def scanner_from_LPS(self) -> geo.FrameTransform:
        return self.LPS_from_scanner.inverse()

    @property
    def world_from_ring(self) -> geo.FrameTransform:
        return (
            self.world_from_patient_array
            @ self.patient_array_from_scanner
            @ self.scanner_from_LPS
            @ self.LPS_from_ring
        )

    @property
    def ring_from_world(self) -> geo.FrameTransform:
        return self.world_from_ring.inverse()

    @property
    def scanner_from_patient_array(self) -> geo.FrameTransform:
        """The transfrom to the scanner (front or back) from the patient array.

        This is the same object as "reference," but from a different ROM file.

        Raises:
            ValueError: If the acquisition is not navigated.
        """
        if not self.navigated:
            raise ValueError("This is not a navigated scan.")

        return geo.FrameTransform(self.patient_to_scanner_matrix)

    @property
    def patient_array_from_scanner(self) -> geo.FrameTransform:
        """The transform to the patient array from the scanner (front or back)."""
        return self.scanner_from_patient_array.inverse()

    @property
    def patient_array_from_loopx(self) -> geo.FrameTransform:
        """The transform to the patient array from LoopX.

        This is the same as the scanner_from_patient_array, but with the axes
        negated.
        """
        return self.patient_array_from_scanner @ self.scanner_from_loopx

    @property
    def index_from_patient_array(self) -> geo.CameraProjection:
        if not self.navigated:
            raise ValueError("This is not a navigated scan.")

        return self.index_from_scanner @ self.scanner_from_patient_array

    @property
    def index_from_world(self) -> geo.CameraProjection:
        return self.index_from_patient_array @ self.patient_array_from_world

    @property
    def LPS_from_patient_array(self) -> geo.FrameTransform:
        if not self.navigated:
            raise ValueError("Not a navigated scan.")
        return self.LPS_from_scanner @ self.scanner_from_patient_array

    @property
    def patient_array_from_LPS(self) -> geo.FrameTransform:
        return self.LPS_from_patient_array.inverse()

    @property
    def RAS_from_patient_array(self) -> geo.FrameTransform:
        return geo.RAS_from_LPS @ self.LPS_from_patient_array

    @property
    def patient_array_from_RAS(self) -> geo.FrameTransform:
        return self.RAS_from_patient_array.inverse()

    @property
    def principle_ray_in_world(self) -> geo.FrameTransform:
        return self.index_from_world.principle_ray_in_world

    @property
    def source_in_world(self) -> geo.Point3D:
        self.scanner_from_patient_array
        return self.world_from_ring @ self.source_in_ring

    @property
    def detector_origin_in_world(self) -> geo.Point3D:
        return self.world_from_ring @ self.detector_origin_in_ring

    def reposition(self, world_from_patient_array: geo.FrameTransform):
        self.world_from_patient_array = geo.f(world_from_patient_array)

    def get_mesh_in_world(self) -> pv.PolyData:
        """Get a mesh of the detector in world coordinates."""
        # Get the corners of the detector
        ul = np.array(self.detector_origin_in_ring)
        ur = np.array(
            self.detector_origin_in_ring
            + self.col_direction_in_ring * self.sensor_width * self.pixel_size[1]
        )
        bl = np.array(
            self.detector_origin_in_ring
            + self.row_direction_in_ring * self.sensor_height * self.pixel_size[0]
        )
        br = np.array(
            self.detector_origin_in_ring
            + self.col_direction_in_ring * self.sensor_width * self.pixel_size[1]
            + self.row_direction_in_ring * self.sensor_height * self.pixel_size[0]
        )
        s = np.array(self.source_in_ring)

        mesh = (
            pv.Sphere(center=s, radius=10)
            + pv.Line(s, ul)
            + pv.Line(s, ur)
            + pv.Line(s, bl)
            + pv.Line(s, br)
            + pv.Line(ul, ur)
            + pv.Line(ur, br)
            + pv.Line(br, bl)
            + pv.Line(bl, ul)
        )

        mesh.transform(np.array(self.world_from_ring), inplace=True)
        return mesh

    @classmethod
    def from_timestamp(cls, data_dir: Path, timestamp: str) -> Acquisition:
        """Load the acquisition from a timestamp.

        Args:
            data_dir: Directory containing the DICOM files.
            timestamp: Timestamp of the acquisition.

        Returns:
            The acquisition.

        """
        device_time = datetime.fromisoformat(timestamp)
        path = dicom_utils.find_dicom(data_dir, device_time)
        return cls(path)

    """Don't use the following. Instead, do object_in_index.backproject(index_from_world)."""

    @overload
    def on_detector_in_ring(self, object_in_index: geo.Point2D) -> geo.Point3D:
        ...

    @overload
    def on_detector_in_ring(self, object_in_index: geo.Line2D) -> geo.Line3D:
        ...

    def on_detector_in_ring(self, object_in_index):
        """Get the 3D representation of the object on the detector plane in the Ring frame.

        Note: index space is (col, row).

        Args:
            q: A 2D point or line in index space.

        Returns:
            The 3D representation of `q` in the Ring frame.

        """
        if isinstance(object_in_index, geo.Point2D):
            return (
                self.detector_origin_in_ring
                + object_in_index.x * self.col_direction_in_ring * self.pixel_size[1]
                + object_in_index.y * self.row_direction_in_ring * self.pixel_size[0]
            )
        elif isinstance(object_in_index, geo.Line2D):
            p1, p2 = object_in_index.as_points()
            q1, q2 = self.on_detector_in_ring(p1), self.on_detector_in_ring(p2)
            return geo.line(q1, q2)
        else:
            raise TypeError(f"Expected Point2D or Line2D, got {type(object_in_index)}")

    @overload
    def on_detector(self, object_in_index: geo.Point2D) -> geo.Point3D:
        ...

    @overload
    def on_detector(self, object_in_index: geo.Line2D) -> geo.Line3D:
        ...

    def on_detector(self, object_in_index):
        """Get the 3D representation of the object on the detector plane in the world frame.

        Note: index space is (col, row).

        Args:
            q: A 2D point or line in index space.

        Returns:
            The 3D representation of `q` in the world frame.

        """
        return self.world_from_ring @ self.on_detector_in_ring(object_in_index)

    @overload
    def backproject_in_ring(self, object_in_index: geo.Point2D) -> geo.Line3D:
        ...

    @overload
    def backproject_in_ring(self, object_in_index: geo.Line2D) -> geo.Plane:
        ...

    def backproject_in_ring(self, object_in_index):
        """Backproject the 2D object through the acquisition source.

        Args:
            object_in_index: A 2D point or line in index space.

        Returns:
            The 3D representation of the backprojected object in the Ring frame.
        """
        q = self.on_detector_in_ring(object_in_index)
        return q.join(self.source_in_ring)

    @overload
    def backproject(self, object_in_index: geo.Point2D) -> geo.Line3D:
        ...

    @overload
    def backproject(self, object_in_index: geo.Line2D) -> geo.Plane:
        ...

    def backproject(self, object_in_index):
        """Backproject the 2D object through the acquisition source.

        Args:
            object_in_index: A 2D point or line in index space.

        Returns:
            The 3D representation of the backprojected object in the "world" frame.
        """
        if isinstance(object_in_index, geo.Point2D):
            return object_in_index.backproject(self.index_from_world)
        elif isinstance(object_in_index, geo.Line2D):
            return object_in_index.backproject(self.index_from_world)
        else:
            raise TypeError(f"Expected Point2D or Line2D, got {type(object_in_index)}")


def load_navigated_ct(
    image_dir: Path,
    device_timestamp: str,
    max_difference: int = 2,
) -> Tuple[Optional[Volume], Optional[geo.F]]:
    """Load the navigated CT volume.

    Places the resulting volume such that the "world" frame is the patient array. In general, when
    using DeepDRR with Loop-X navigated scans, always choose the "world" frame as the patient array.

    Args:
        image_dir: Directory containing DICOM files.

    Returns:
        The navigated CT volume, where the "world" frame is the patient array.

    """

    try:
        device_time = datetime.fromisoformat(device_timestamp)
    except ValueError:
        # log.warning(f"Could not parse timestamp {device_timestamp}.\n")
        return None, None

    # First, find the CT based on the device time.
    # pattern = str(
    #     image_dir.resolve() / "**" / f"{device_time:%Y-%m-%d.%H%M}[0-9][0-9][0-9][0-9][0-9]/"
    # )
    pattern = f"**/{device_time:%Y-%m-%d.%H%M}[0-9][0-9][0-9][0-9][0-9]/"
    possible_dirs = list(image_dir.glob(pattern))

    if not possible_dirs:
        log.warning(f"Could not find CT scan for device time {device_time}.\n")
        return None, None

    scan_times = [datetime.strptime(d.name[:-3], "%Y-%m-%d.%H%M%S") for d in possible_dirs]
    diffs = [(t - device_time).seconds for t in scan_times]
    idx = np.argmin(diffs)
    diff = diffs[idx]
    if diff > max_difference:
        log.warning(f"Found scan {diff} seconds away from device time {device_time}.")
        return None, None

    dicom_dir = possible_dirs[idx]

    # Now, load the first slice to get the metadata.
    first_slice = list(dicom_dir.glob("*.dcm"))[0]
    ds = pydicom.dcmread(first_slice)

    # This provides world_from_anatomical for the Nifti file, since DeepDRR prefers everything to be
    # in RAS and Nifti.
    if 0x30BB2050 not in ds:
        log.warning(f"Not a navigated CT: {first_slice}")
        return None, None
    calibration_matrix_front = np.array(ds[0x30BB2051].value).reshape(4, 4)
    patient_to_scanner_matrix = np.array(ds[0x30BB2055].value).reshape(4, 4)
    LPS_from_scanner = geo.F(calibration_matrix_front)
    scanner_from_patient_array = geo.F(patient_to_scanner_matrix)
    patient_array_from_RAS = (
        geo.RAS_from_LPS @ LPS_from_scanner @ scanner_from_patient_array
    ).inverse()
    # patient_array_from_RAS = LPS_from_scanner @ scanner_from_patient_array
    patient_array_from_loopx = scanner_from_patient_array.inverse() @ Acquisition.scanner_from_loopx

    nifti_dir = dicom_dir.parent / f"nifti_{dicom_dir.name}"
    if nifti_dir.exists() and list(nifti_dir.glob("*.nii.gz")):
        pass
    else:
        # Convert the DICOMs to Nifti.
        if nifti_dir.exists():
            rmtree(nifti_dir)
        nifti_dir.mkdir(exist_ok=True, parents=True)
        log.info(f"Converting {dicom_dir} to Nifti.")
        args = [
            "dcm2niix",
            "-z",
            "y",
            "-o",
            str(nifti_dir),
            str(dicom_dir),
        ]
        print(f"Running\n{' '.join(args)}")
        # log.debug(f"Running {' '.join(args)}")
        log.info(
            f"Ensure dcm2niix is installed and up-to-date. (Older versions will not convert Loop-X scans, which have a gantry tilt.\n"
            f"Download the latest version: `curl -fLO https://github.com/rordenlab/dcm2niix/releases/latest/download/dcm2niix_lnx.zip`\n"
            f"And unzip somewhere on your PATH: `unzip dcm2niix_lnx.zip`"
        )
        sp.run(args)

    # Load the Nifti.
    nifti_path = list(nifti_dir.glob("*.nii.gz"))[0]
    volume = Volume.from_nifti(
        nifti_path,
        world_from_anatomical=patient_array_from_RAS,
        use_thresholding=True,
    )
    return volume, patient_array_from_loopx
