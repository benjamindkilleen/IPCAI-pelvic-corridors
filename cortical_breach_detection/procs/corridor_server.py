from collections import deque
import logging
import math
import multiprocessing as mp
import subprocess as sp
from pathlib import Path
from re import M
from typing import Any, Dict, List, Optional, Tuple
import deepdrr
from deepdrr.utils import image_utils
from matplotlib import pyplot as plt
import zmq
import cv2
import time
import numpy as np
from deepdrr import geo
import traceback
from datetime import datetime
import torch
import pydicom
import torchvision.transforms.functional as TF
from rich.logging import RichHandler
import pyvista as pv
from deepdrr.annotations import LineAnnotation

from ..models import UNet
from ..loopx import DEGREE_SIGN, Acquisition, LoopX, load_navigated_ct
from ..threads import ImagePublisher, PosePublisher
from .. import utils
from ..utils import combine_heatmap, eval_utils, onedrive_utils, cv_utils
from ..viewpoint_planning import plan_viewpoint

log = logging.getLogger(__name__)


class CorridorServer(mp.Process):
    # 4x4 arrays indexed by (to_frame, from_frame)
    # hlworld_from_reference_deque:

    # Same as above, indexed by the isoformat timestamp
    tracker_frames: Dict[str, Dict[Tuple[str, str], geo.FrameTransform]]

    # STL file is stored in LPS coordinates, remember
    pelvis_model_from_pelvis_marker = geo.f(
        """-0.91108588 -0.39857756 -0.13363132 25.74871523
        -0.21853024 0.17657402 0.96325751 -43.43864193
        -0.35911828 0.90374578 -0.24713664 -153.38330693
        0.00000000 0.00000000 0.00000000 1.00000000"""
    )
    pelvis_model_from_pelvis_marker_RAS = (
        geo.RAS_from_LPS @ pelvis_model_from_pelvis_marker @ geo.LPS_from_RAS
    )

    pointer_from_pointer_back = geo.f(
        """1	0	0	-11.7029
        0	-1	-1.22465e-16	9.92828
        0	1.22465e-16	-1	-83.6439
        0	0	0	1"""
    )

    pointer_from_pointer_tip = geo.f(
        """0.836488	0	0.547985	140.63
        -0.0836032	0.988293	0.127619	32.7509
        -0.54157	-0.152565	0.826696	212.156
        0	0	0	1"""
    )

    pelvis_markers = np.fromstring(
        """
        0 0 0
        9.55 29.50 -5.39
        -13.36 35.97 -13.29
        -54.79 3.28 -14.01
        """,
        sep=" ",
    ).reshape(-1, 3)

    def __init__(
        self,
        queue: mp.Queue,
        loopx_app_queue: mp.Queue,
        image_publisher_queue: mp.Queue,
        pose_publisher_queue: mp.Queue,
        hololens_server_queue: mp.Queue,
        image_publisher: Dict[str, Any] = {},
        pose_publisher: Dict[str, Any] = {},
        device: Dict[str, Any] = {},
        projector: Dict[str, Any] = {},
        onedrive_dir: str = "~/datasets/OneDrive",
        xray_dir: str = "datasets/2022-10-13_corridor-triangulation_phantom",
        phantom_dir: str = "datasets/2022_AR-Fluoroscopy/2022-08-17_LFOV-CT",
        checkpoint: Optional[str] = None,
        model: Dict[str, Any] = {},
        xi: float = 0.0,
        image_size: List[int] = [512, 512],
        monitor: bool = True,
        left_side: bool = True,
        show_scene: bool = True,
        resync: bool = False,
        adjust_corridor: bool = True,
        adjustment_radius: float = 13,
    ):
        """Corridor server.

        Args:
            queue (mp.Queue): Queue for receiving data (called the corridor_server_queue elsewhere).
            loopx_app_queue (mp.Queue): Queue for sending data to LoopX app.
            pose_publisher_queue (mp.Queue): Queue for sending data to the pose publisher.
            image_publisher_queue (mp.Queue): Queue for sending data to the image publisher.
            pose_publisher (Dict[str, Any]): Pose publisher configuration.
            image_publisher (Dict[str, Any]): Image publisher configuration.
            device (Dict[str, Any], optional): Device configuration for the Loop-X. Defaults to {}.
            xray_dir (Path): Path to the OneDrive monitored folder where raw DICOMs will be dropped, from Loop-X export.
            phantom_dir (Path): Relative path to the OneDrive folder with the Phantom CT
            checkpoint (Optional[str], optional): Path to the checkpoint. Defaults to None.
            model (Dict[str, Any], optional): Model configuration. Defaults to {}.
            xi (float, optional): Angulation about the corridor for the second image, in degrees. Defaults to 0.0.
            image_size (List[int], optional): Image size for feeding into the model. Defaults to [512, 512].
            monitor (bool, optional): Whether to monitor the OneDrive folder for new DICOMs. Defaults to True.
            left_side (bool, optional): Whether operating on patient left. Defaults to True.
            show_scene (bool, optional): Whether to show the scene in a pyvista window (if possible). Defaults to True.
            resync (bool, optional): Whether to resync the tracker frames. Defaults to False.
            adjust_corridor (bool, optional): Whether to adjust the corridor using nonlinear optimization of the reprojection. Defaults to True.
            adjustment_radius (float, optional): Radius of the corridor to project when adjusting to maximize posterior. Defaults to 13.

        """
        super().__init__()
        self.queue = queue
        self.loopx_app_queue = loopx_app_queue
        self.image_publisher_queue = image_publisher_queue
        self.pose_publisher_queue = pose_publisher_queue
        self.hololens_server_queue = hololens_server_queue
        self.image_publisher_config = image_publisher
        self.pose_publisher_config = pose_publisher
        self.device_config = device
        self.projector_config = projector
        self.onedrive_dir = Path(onedrive_dir).expanduser()
        self.xray_reldir = xray_dir
        self.xray_dir = self.onedrive_dir / xray_dir
        self.phantom_reldir = phantom_dir
        self.phantom_dir = self.onedrive_dir / phantom_dir
        self.checkpoint = checkpoint
        self.model_config = model
        self.xi = math.radians(xi)
        self.image_size = deepdrr.utils.listify(image_size, 2)
        self.monitor = monitor
        self.left_side = left_side
        self.do_show_scene = show_scene
        self.resync = resync
        self.adjust_corridor = adjust_corridor
        self.adjustment_radius = adjustment_radius

        self.onedrive = onedrive_utils.OneDrive(self.onedrive_dir)

        self.images_dir = Path("images")
        self.images_dir.mkdir(exist_ok=True)
        self.info_dir = Path("info")
        self.info_dir.mkdir(exist_ok=True)

        self.tracker_frames = {}
        self.first_image_success = False
        self.hlworld_from_reference_deque = deque(maxlen=5)
        self.publish_hl_poses = True

        # None until first detected
        self.patient_array_from_corridor = None
        self.patient_array_from_kwire = None
        self.acquisition_1: Optional[Acquisition] = None
        self.acquisition_2: Optional[Acquisition] = None

        # None until ct timestamp provided
        self.ct: Optional[deepdrr.Volume] = None
        self.projector: Optional[deepdrr.Projector] = None

    def get_hlworld_from_reference(self) -> Optional[np.ndarray]:
        """Get the average of the last `maxlen` HLWorldFromReference matrices."""
        if len(self.hlworld_from_reference_deque) == 0:
            return None

        # def reject_outliers(data, m=2):
        #     return data[abs(data - np.mean(data, axis=0)) < m * np.std(data, axis=0)]

        hlworld_from_reference = np.stack(self.hlworld_from_reference_deque)
        return geo.frame_transform(np.mean(hlworld_from_reference, axis=0))

    def download_phantom(self, resync: bool = False):
        self.onedrive.download(self.phantom_reldir, resync=resync)

    def start_monitor(self):
        """Monitor the OneDrive folder for new images."""
        # self.download_phantom()
        self.onedrive.monitor(self.xray_reldir)

    def run(self):
        """Run the corridor server."""

        logging.basicConfig(
            level=logging.DEBUG,
            handlers=[RichHandler(rich_tracebacks=True)],
        )

        # Start the monitor
        if self.monitor:
            self.start_monitor()

        self.image_publisher = ImagePublisher(
            self.image_publisher_queue, **self.image_publisher_config
        )
        self.image_publisher.start()

        self.pose_publisher = PosePublisher(self.pose_publisher_queue, **self.pose_publisher_config)
        self.pose_publisher.start()

        self.device = LoopX(**self.device_config)

        # For using the phantom (deprecated)
        # phantom_path = self.phantom_dir / "2 LFOV_FullScan (2022-08-17 2239).nii.gz"
        # phantom_bone_path = self.phantom_dir / "bone.nii.gz"
        # if not phantom_path.exists() or not phantom_bone_path.exists():
        #     self.download_phantom(resync=True)

        # # For debugging, try another CT
        # self.ct = deepdrr.Volume.from_nifti(
        #     phantom_path,
        #     materials=dict(bone=phantom_bone_path),
        # )
        # self.projector = deepdrr.Projector(self.ct, device=self.device, **self.projector_config)
        # self.projector.initialize()

        if self.checkpoint is None:
            log.warning("No checkpoint provided. Model is untrained.")
            self.model = UNet(**self.model_config)
        else:
            self.model = UNet.load_from_checkpoint(self.checkpoint)

        # self.model.to("cuda:1")
        # self.model.cuda()
        self.model.eval()

        self.running = True
        last_publish_time = time.time()

        while self.running:
            try:
                # Check the queue for a new item.
                item = self.queue.get()
                if item is None:
                    log.warning("Corridor server received None.")
                    time.sleep(0.1)
                    continue

                message_type = item[0]
                if message_type == "tracker_frame":
                    # Received a frame from the Tracker Laptop or HoloLens.
                    # Format: (message_type, to_frame, from_frame, transform)
                    to_frame, from_frame, transform, timestamp = item[1:5]
                    if timestamp not in self.tracker_frames:
                        self.tracker_frames[timestamp] = {}
                    if (to_frame, from_frame) not in self.tracker_frames[timestamp]:
                        self.tracker_frames[timestamp][(to_frame, from_frame)] = geo.F(transform)
                        self.tracker_frames[timestamp][(from_frame, to_frame)] = geo.F(
                            transform
                        ).inv
                        log.info(f"Received '{timestamp}': '{to_frame} -> {from_frame}' transform.")
                elif message_type == "hololens_frame":
                    # Received a frame from the HoloLens.
                    # Format: (message_type, to_frame, from_frame, transform)
                    to_frame, from_frame, transform = item[1:4]
                    if self.publish_hl_poses:
                        log.info(f"Received hololens frame '{to_frame} -> {from_frame}'")
                    if to_frame == "hlworld" and from_frame == "reference":
                        # TODO: do some averaging here, based on timestamps?
                        # Or maybe that should be in the hololens server.
                        self.hlworld_from_reference_deque.append(geo.F(transform).as_quatpos())
                        if time.time() - last_publish_time > 0.2 and self.publish_hl_poses:
                            log.info("Publishing pose.")
                            hlworld_from_reference = self.get_hlworld_from_reference()
                            self.publish_corridor(hlworld_from_reference)
                            self.publish_kwire(hlworld_from_reference)
                            self.publish_loopx(hlworld_from_reference)
                            last_publish_time = time.time()
                    else:
                        log.warning(f"Unknown hololens frame '{to_frame} -> {from_frame}'")
                elif message_type == "hl_toggle":
                    self.publish_corridor()
                    self.publish_kwire()
                    self.publish_loopx()
                    self.hololens_server_queue.put(("toggle",))
                    self.publish_hl_poses = not self.publish_hl_poses
                    log.info(f"Publishing HL poses: {self.publish_hl_poses}")
                elif message_type == "ct_timestamp":
                    # Format: (message_type, ct_timestamp)
                    if self.projector is not None:
                        # Clear the previous projector
                        self.projector.free()
                        self.projector = None
                        self.ct = None

                    ct_timestamp = item[1]
                    log.info(f"Looking for CT {ct_timestamp} in {self.xray_dir}")
                    self.ct, patient_array_from_loopx = load_navigated_ct(
                        self.xray_dir, ct_timestamp
                    )
                    success = self.ct is not None
                    if success:
                        self.projector = deepdrr.Projector(
                            self.ct, device=self.device, **self.projector_config
                        )
                        self.projector.initialize()
                        self.show_scene_cadaver(patient_array_from_loopx=patient_array_from_loopx)
                    self.loopx_app_queue.put(dict(success=success))
                elif message_type == "first_image":
                    # Receive user input for the first image.
                    # Format: (message_type, loopx_pose, device_timestamp)
                    loopx_pose = item[1]
                    device_timestamp = item[2]

                    log.info(f"Image 1: '{device_timestamp}'")

                    msg = self.run_first_image(loopx_pose, device_timestamp)
                    self.loopx_app_queue.put(msg)
                    log.info(f"Done analyzing first image: '{device_timestamp}'")
                elif message_type == "second_image":
                    # Receive user input for the second image.
                    # Format: (message_type, device_timestamp)
                    loopx_pose = item[1]
                    device_timestamp = item[2]

                    log.info(f"Image 2: '{device_timestamp}'")

                    self.run_second_image(loopx_pose, device_timestamp)
                    log.info(f"Done analyzing second image: '{device_timestamp}'")
                elif message_type == "update_xi":
                    # Format: (message_type, xi (degrees))
                    xi = item[1]
                    if xi is not None:
                        self.xi = math.radians(xi)
                        log.info(f"Updated xi: {self.xi} ({xi}{DEGREE_SIGN})")
                elif message_type == "update_adjustment_radius":
                    adjustment_radius = item[1]
                    if adjustment_radius is not None:
                        self.adjustment_radius = adjustment_radius
                        log.info(f"Updated adjustment_radius: {self.adjustment_radius}")
                elif message_type == "switch_side":
                    # Format: (message_type,)
                    self.left_side = not self.left_side
                    log.info(f"Switched side to left_side={self.left_side}")
                elif message_type == "switch_show_scene":
                    self.do_show_scene = not self.do_show_scene
                    log.info(f"Switched show_scene to {self.do_show_scene}")
                elif message_type == "switch_adjust_corridor":
                    self.adjust_corridor = not self.adjust_corridor
                    log.info(f"Switched adjust_corridor to {self.adjust_corridor}")
                elif message_type == "close":
                    self.running = False
                    log.info("Closing corridor server.")
                else:
                    log.warning(f"Corridor server received unknown message: {item}")

            except KeyboardInterrupt:
                self.running = False
                break
            except Exception as e:
                if message_type == "first_image":
                    self.loopx_app_queue.put(dict(success=False))
                log.error(e)
                traceback.print_exc()
                continue

        self.onedrive.terminate()
        if self.projector is not None:
            self.projector.free()

    def get_closest_image(
        self, device_timestamp: str, max_difference: Optional[int] = 3
    ) -> Tuple[Optional[Path], Optional[datetime]]:
        """Get the image closest to the given timestamp.

        Args:
            device_timestamp (str): Timestamp as read from the Loop-X of the image in question.
            max_difference (int, optional): Maximum difference in seconds between the given timestamp and the closest image. Defaults to None.

        Returns:
            Path: _description_
        """

        # Get the image path
        try:
            device_time = datetime.fromisoformat(device_timestamp)
        except ValueError:
            # log.warning(f"Could not parse timestamp {device_timestamp}.\n")
            return None, None

        image_paths: List[Path] = []
        device_times: List[datetime] = []
        for p in self.xray_dir.glob("**/*.dcm"):
            try:
                t = datetime.strptime(p.stem.split("_")[1][:-3], r"%Y%m%d%H%M%S")
            except:
                # log.debug(f"Could not parse timestamp from {p}.")
                continue

            image_paths.append(p)
            device_times.append(t)

        if len(image_paths) == 0 or len(device_times) == 0:
            return None, None

        assert len(image_paths) == len(device_times)
        image_diffs = [abs((device_time - t).total_seconds()) for t in device_times]
        image_path_idx = np.argmin(image_diffs)
        image_path = image_paths[image_path_idx]
        dt = device_times[image_path_idx]
        if max_difference is not None and image_diffs[image_path_idx] > max_difference:
            log.debug(
                f"Image difference too large: {image_diffs[image_path_idx]} for {device_timestamp}, {image_path} with parsed time {dt}"
            )
            return None, None

        return image_path, dt

    def load_image(self, image_path: Path) -> np.ndarray:
        image = pydicom.dcmread(image_path).pixel_array
        image = image.astype(np.float32)
        image = image / (2**16 - 1)
        return image

    def draw_corridor(
        self, image: np.ndarray, startpoint: np.ndarray, direction: np.ndarray, hip_mask: np.ndarray
    ) -> np.ndarray:
        """Draw the corridor on the image.

        Args:
            image (np.ndarray): The image to draw on.
            startpoint (np.ndarray): The startpoint of the corridor.
            direction (np.ndarray): The direction of the corridor.
            hip_mask (np.ndarray): The hip mask.

        Returns:
            np.ndarray: The image with the corridor drawn on it.
        """

        corridor_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        x0, y0 = startpoint
        x1, y1 = startpoint + direction * 10000

        # Draw the direction
        corridor_image = cv2.line(
            corridor_image,
            (int(x0), int(y0)),
            (int(x1), int(y1)),
            (255, 255, 255),
            thickness=5,
        )

        # Draw the hip mask
        # corridor_image[hip_mask < 0.5] = [0, 0, 0]
        return corridor_image[:, :, 0]

    def save_images(self, base: str, **kwargs) -> Dict[str, Path]:
        """Save the images and info to the current images directory.

        TODO: Organize output images by "run," so that that corresponding 1st and 2nd images are in the same folder.

        """

        paths = {}

        for k, v in kwargs.items():
            if v is None:
                continue
            p = self.images_dir / f"{base}_{k}.png"
            image_utils.save(p, v)
            paths[k] = p
        return paths

    def show_scene_cadaver(
        self,
        *meshes: pv.PolyData,
        patient_array_from_loopx: geo.F,
    ):
        if not self.do_show_scene:
            return
        plotter = pv.Plotter()
        plotter.set_background("white")
        for mesh in meshes:
            plotter.add_mesh(mesh, color="black")

        if self.ct is not None:
            ct_mesh = self.ct.get_mesh_in_world(full=True, use_cached=True)
            plotter.add_mesh(ct_mesh, color="white")

        loopx_markers = pv.PolyData()
        for i in range(LoopX.markers.shape[0]):
            loopx_markers += pv.Sphere(
                center=np.array(patient_array_from_loopx @ geo.p(LoopX.markers[i])),
                radius=5,
            )
        plotter.add_mesh(loopx_markers, color="yellow")

        patient_array_markers = pv.PolyData()
        for i in range(Acquisition.markers_in_patient_array.shape[0]):
            patient_array_markers += pv.Sphere(
                center=Acquisition.markers_in_patient_array[i], radius=5
            )
        plotter.add_mesh(patient_array_markers, color="gray")

        origin = pv.Line(np.array([0, 0, 0]), np.array([20, 0, 0]))
        origin += pv.Line(np.array([0, 0, 0]), np.array([0, 20, 0]))
        origin += pv.Line(np.array([0, 0, 0]), np.array([0, 0, 20]))
        plotter.add_mesh(origin, color="magenta")

        if self.acquisition_1 is not None:
            acquisition_1_mesh = self.acquisition_1.get_mesh_in_world()
            plotter.add_mesh(acquisition_1_mesh, color="blue")
        if self.acquisition_2 is not None:
            acquisition_2_mesh = self.acquisition_2.get_mesh_in_world()
            plotter.add_mesh(acquisition_2_mesh, color="red")

        plotter.show()

    def show_scene(
        self,
        frames: Dict[Tuple[str, str], geo.F],
        *meshes: pv.PolyData,
        acquisition: Optional[Acquisition] = None,
    ):
        """Show the scene, with everything in the world (patient) frame."""
        if not self.do_show_scene:
            return
        plotter = pv.Plotter()
        plotter.set_background("white")
        for mesh in meshes:
            plotter.add_mesh(mesh, color="black")

        if acquisition is not None:
            patient_array_from_loop_x = (
                acquisition.patient_array_from_scanner @ acquisition.scanner_from_loopx
            )
        else:
            patient_array_from_loop_x = self.device.world_from_loopx

        patient_array_from_pointer = (
            Acquisition.patient_array_from_reference
            @ frames[("reference", "tracker")]
            @ frames[("tracker", "pointer")]
        )
        patient_array_from_pelvis_model = (
            Acquisition.patient_array_from_reference
            @ frames[("reference", "tracker")]
            @ frames[("tracker", "pelvis_marker")]
            @ self.pelvis_model_from_pelvis_marker.inv
        )
        startpoint_in_pelvis_model_lps, endpoint_in_pelvis_model_lps = utils.load_line_markup(
            self.phantom_dir / "trajectory_left.mrk.json"
            if self.left_side
            else self.phantom_dir / "trajectory_right.mrk.json"
        )
        startpoint_in_pelvis_model = geo.LPS_from_RAS @ startpoint_in_pelvis_model_lps
        endpoint_in_pelvis_model = geo.LPS_from_RAS @ endpoint_in_pelvis_model_lps
        pelvis_mesh = pv.read(self.phantom_dir / "PelvisPhantomModel.stl")
        pelvis_mesh = pelvis_mesh.transform(np.array(geo.LPS_from_RAS))
        pelvis_mesh = pelvis_mesh.transform(np.array(patient_array_from_pelvis_model))
        plotter.add_mesh(pelvis_mesh, color="white")

        pointer_mesh = pv.PolyData()
        pointer_mesh += pv.Sphere(
            center=np.array(patient_array_from_pointer @ self.pointer_from_pointer_tip.o),
            radius=5,
        )
        pointer_mesh += pv.Sphere(
            center=np.array(patient_array_from_pointer @ self.pointer_from_pointer_back.o),
            radius=5,
        )
        pointer_mesh += pv.Line(
            np.array(patient_array_from_pointer @ self.pointer_from_pointer_tip.o),
            np.array(patient_array_from_pointer @ self.pointer_from_pointer_back.o),
        )
        plotter.add_mesh(pointer_mesh, color="gray")

        loopx_markers = pv.PolyData()
        for i in range(LoopX.markers.shape[0]):
            loopx_markers += pv.Sphere(
                center=np.array(patient_array_from_loop_x @ geo.p(LoopX.markers[i])),
                radius=5,
            )
        plotter.add_mesh(loopx_markers, color="yellow")

        origin = pv.Line(np.array([0, 0, 0]), np.array([20, 0, 0]))
        origin += pv.Line(np.array([0, 0, 0]), np.array([0, 20, 0]))
        origin += pv.Line(np.array([0, 0, 0]), np.array([0, 0, 20]))
        plotter.add_mesh(origin, color="magenta")

        corridor_mesh = pv.PolyData()
        # corridor_mesh += pv.Line(
        #     np.array(patient_array_from_pelvis_model @ startpoint_in_pelvis_model), np.array(patient_array_from_pelvis_model @ endpoint_in_pelvis_model)
        # )
        corridor_mesh += pv.Sphere(
            center=np.array(patient_array_from_pelvis_model @ startpoint_in_pelvis_model), radius=5
        )
        corridor_mesh += pv.Sphere(
            center=np.array(patient_array_from_pelvis_model @ endpoint_in_pelvis_model), radius=5
        )
        plotter.add_mesh(corridor_mesh, color="green")

        if self.acquisition_1 is not None:
            acquisition_1_mesh = self.acquisition_1.get_mesh_in_world()
            plotter.add_mesh(acquisition_1_mesh, color="blue")
        if self.acquisition_2 is not None:
            acquisition_2_mesh = self.acquisition_2.get_mesh_in_world()
            plotter.add_mesh(acquisition_2_mesh, color="red")

        screenshot = plotter.show()
        return screenshot

    def analyze_image(self, image: np.ndarray) -> Tuple[np.ndarray, ...]:
        log.info(f"Running model on image...")
        image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(self.model.device)
        image_tensor = TF.resize(image_tensor, self.image_size)
        left_side = np.array([not self.left_side], np.int32)  # TODO: model flipped for some reason
        (
            _,
            kwire_mask_tensor,
            hip_mask_tensor,
            startpoint_heatmap_tensor,
            startpoint_depthmap_tensor,
            corridor_mask_tensor,
        ) = self.model.process_outputs(image_tensor, left_side, original_image_size=image.shape)
        kwire_mask = utils.get_numpy(kwire_mask_tensor)[0]
        hip_mask = utils.get_numpy(hip_mask_tensor)[0]
        startpoint_heatmap = utils.get_numpy(startpoint_heatmap_tensor)[0, 0]
        startpoint_depthmap = utils.get_numpy(startpoint_depthmap_tensor)[0, 0]
        corridor_mask = utils.get_numpy(corridor_mask_tensor)[0, 0]
        log.info(f"Done running model on image.")

        return kwire_mask, hip_mask, startpoint_heatmap, startpoint_depthmap, corridor_mask

    def check_tracker_frames(self, frames) -> bool:
        """Check if the tracking is good.

        Returns:
            bool: True if the tracking is good.
        """
        # Check that we have all the necessary frames.
        # TODO; add timestamps to frames, check if close enough to image.
        frames_needed = [
            ("tracker", "loop_x"),
            ("reference", "tracker"),
            ("tracker", "pelvis_marker"),
            ("tracker", "pointer"),
        ]
        tracking_is_good = True
        for to_frame, from_frame in frames_needed:
            if (to_frame, from_frame) not in frames:
                log.warning(f"Missing frame '{to_frame} -> {from_frame}'.")
                tracking_is_good = False

        return tracking_is_good

    def check_hololens_frames(self) -> bool:
        frames_needed = [
            ("hlworld", "reference"),
        ]
        tracking_is_good = True
        for to_frame, from_frame in frames_needed:
            if (to_frame, from_frame) not in self.hololens_frames:
                log.warning(f"Missing frame {to_frame} -> {from_frame} from Hololens.\n")
                tracking_is_good = False
        return tracking_is_good

    def publish_loopx(self, hlworld_from_reference: Optional[geo.F] = None) -> bool:
        """Publish the loopx pose to hololens.

        Returns:
            bool: True if the pose was published, False if not.
        """
        if not self.publish_hl_poses:
            return False

        if hlworld_from_reference is None:
            hlworld_from_reference = self.get_hlworld_from_reference()
            if hlworld_from_reference is None:
                return False

        hlworld_from_source = np.array(
            hlworld_from_reference
            @ Acquisition.reference_from_patient_array
            @ self.device.world_from_source
        )
        hlworld_from_detector = np.array(
            hlworld_from_reference
            @ Acquisition.reference_from_patient_array
            @ self.device.world_from_detector
        )
        hlworld_from_gantry = np.array(
            hlworld_from_reference
            @ Acquisition.reference_from_patient_array
            @ self.device.world_from_gantry
        )

        self.pose_publisher_queue.put((3, hlworld_from_source))
        self.pose_publisher_queue.put((4, hlworld_from_detector))
        self.pose_publisher_queue.put((5, hlworld_from_gantry))

        return True

    def publish_corridor(self, hlworld_from_reference: Optional[geo.F] = None) -> bool:
        if not self.publish_hl_poses:
            return False

        if hlworld_from_reference is None:
            hlworld_from_reference = self.get_hlworld_from_reference()
            if hlworld_from_reference is None:
                return False
        if self.patient_array_from_corridor is None:
            return False

        hlworld_from_corridor = (
            hlworld_from_reference
            @ Acquisition.reference_from_patient_array
            @ self.patient_array_from_corridor
        )
        self.pose_publisher_queue.put(
            (6, np.array(hlworld_from_corridor), np.array([1, 1, self.corridor_radius]))
        )
        return True

    def publish_kwire(self, hlworld_from_reference: Optional[geo.F] = None) -> bool:
        if not self.publish_hl_poses:
            return False

        if hlworld_from_reference is None:
            hlworld_from_reference = self.get_hlworld_from_reference()
            if hlworld_from_reference is None:
                return False
        if self.patient_array_from_kwire is None:
            return False

        hlworld_from_kwire = (
            hlworld_from_reference
            @ Acquisition.reference_from_patient_array
            @ self.patient_array_from_kwire
        )
        self.pose_publisher_queue.put((7, np.array(hlworld_from_kwire)))
        return True

    def check_for_pelvis_marker(self, acquisition: Acquisition) -> Optional[geo.FrameTransform]:
        """Check if the pelvis marker is visible in the image.

        If it is, run PnP to obtain the world_from_pelvis_marker transform.

        Returns:
            bool: True if the pelvis marker is visible.
        """
        image = acquisition.image

        # TODO: Check if the pelvis marker is visible by looking for four spheres.
        # 70 approx radius
        # Note, values based on full resolution images.
        circles = cv_utils.detect_circles(
            image,
            min_radius=60,
            max_radius=80,
            min_distance=150,
            blur=0,
        )
        # Could also get the four with highest Y value.
        indices = set(np.argsort(circles[:, 1])[-4:].astype(int))
        if indices != {0, 1, 2, 3}:
            log.warning(f"Four lowest spheres are not highest confidence.")
            return None

        circles = circles[:4]  # I think ough outputs circles sorted by votes.
        image = image_utils.draw_circles(image, circles)
        image_utils.save(f"debug/pelvis_marker.png", image)

        # acquisition
        # TODO: provide an initial guess based on tracking?
        success, r, t = cv2.solvePnP(
            self.pelvis_markers,
            circles[:, :2],
            np.array(acquisition.intrinsic),
            method=cv2.SOLVEPNP_IPPE,
        )

        if not success:
            log.warning("Failed to solve PnP for pelvis marker.")
            return None

        log.debug(f"r: {r}, t: {t}")

        # TODO: compare the output of this to the known position of the markers from tracking.
        # If mathias's theory about the error is correct, they should be off by a fixed amount.
        # This gives us a correction to apply to the pelvis position for the second image?
        # Would be way simpler to just get the reliable tracking info from BrainLab.

        return None

    def run_first_image(
        self,
        loopx_pose: Dict[str, float],
        image_timestamp: str,
    ) -> Dict[str, Any]:
        """Run the first image algorithm, for cadaver study.

        Args:
            loopx_pose: The pose of the LoopX in the world coordinate system.
            image_timestamp: The timestamp of the image.

        Returns:
            A dictionary with the results, to be sent to the loopx_app.
        """

        self.first_image_success = False

        dicom_path, device_time = self.get_closest_image(image_timestamp)
        if dicom_path is None or device_time is None:
            log.warning("No image found.")
            return dict(success=False)

        # Load the image, reposition the device
        self.acquisition_1 = Acquisition(dicom_path)
        self.device.reposition(self.acquisition_1.patient_array_from_loopx, **loopx_pose)

        # Configure virtual device to match acquisition
        image = self.acquisition_1.image
        self.device.sensor_height = image.shape[0]
        self.device.sensor_width = image.shape[1]
        device_config = self.device.get_config()

        if self.ct is not None and self.projector is not None:
            drr = self.projector(self.acquisition_1.index_from_patient_array)
        else:
            drr = None

        # Run the model on the image
        (
            kwire_mask,
            hip_mask,
            startpoint_heatmap,
            startpoint_depthmap,
            corridor_mask,
        ) = self.analyze_image(deepdrr.utils.neglog(image))

        (
            corridor_line_in_index,
            startpoint_in_index,
            direction_in_index,
        ) = eval_utils.locate_corridor(startpoint_heatmap, corridor_mask, mask=hip_mask)
        if (
            corridor_line_in_index is None
            or startpoint_in_index is None
            or direction_in_index is None
        ):
            log.warning("Could not locate corridor.")
            return dict(success=False)

        # Sanity check on startpoint
        if (
            startpoint_in_index[1] < 0
            or startpoint_in_index[1] >= image.shape[0]
            or startpoint_in_index[0] < 0
            or startpoint_in_index[0] >= image.shape[1]
        ):
            log.warning(f"Startpoint is outside the image: {startpoint_in_index}")
            return dict(success=False)

        # Estimate the 3D startpoint
        d = startpoint_depthmap[int(startpoint_in_index[1]), int(startpoint_in_index[0])]
        if d > 0.5:
            log.warning(
                f"Startpoint depth {d} is too large. Did you forget which model you're running?"
            )
        f_mm = 1247.0901012420654
        z_cam = d * f_mm
        startpoint_ray_in_camera = geo.v(
            self.device.camera3d_from_index @ startpoint_in_index
        ).hat()
        angle = startpoint_ray_in_camera.angle(geo.v(0, 0, 1))
        desired_isocenter_in_camera = geo.p(0, 0, 0) + startpoint_ray_in_camera * (
            z_cam / math.cos(angle)
        )
        desired_isocenter = self.device.world_from_camera3d @ desired_isocenter_in_camera

        # Run Preuhs algorithm.
        ring_from_preuhs = geo.FrameTransform.from_rt(
            rotation=np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]]),
        )
        index_from_patient_array = self.acquisition_1.index_from_world
        patient_array_from_preuhs = self.acquisition_1.world_from_ring @ ring_from_preuhs
        index_from_preuhs = index_from_patient_array @ patient_array_from_preuhs
        c1 = geo.get_data(startpoint_in_index)
        c2 = geo.get_data(startpoint_in_index + 100 * direction_in_index)
        P1 = geo.get_data(index_from_preuhs)
        ray_preuhs, t_preuhs = plan_viewpoint(c1, c2, P1, self.xi)
        ray_world = patient_array_from_preuhs @ geo.Vector3D(ray_preuhs)
        # Re-align the device to the planned view.
        self.device.align(
            desired_isocenter,
            ray_world,
            ray_angle_tolerance=0,
            detector_to_point_distance=(f_mm - z_cam),
            detector_to_point_tolerance=300,
            traction_yaw_tolerance=15,
            min_gantry_tilt=-20,
            max_gantry_tilt=20,
        )

        if self.projector is not None:
            drr_planned = self.projector()
        else:
            drr_planned = None

        self.images_dir = Path(
            f"{device_time:%Y-%m-%d_%H-%M-%S}_{datetime.now():%Y-%m-%d_%H-%M-%S}"
        )
        self.images_dir.mkdir()
        base = "1st"
        # TODO; IO thread for saving images?
        log.info("Saving images.")
        image_paths = []
        image_paths = self.save_images(
            base,
            image=image,
            drr=drr,
            drr_planned=drr_planned,
            corridor_mask=combine_heatmap(image, corridor_mask),
            # kwire_mask=combine_heatmap(image, kwire_mask > 0.5),
            # hip_mask=combine_heatmap(image, hip_mask > 0.5),
            startpoint_heatmap=combine_heatmap(image, startpoint_heatmap),
            # startpoint_depthmap=combine_heatmap(image, startpoint_depthmap),
        )

        if self.do_show_scene:
            ray_mesh = pv.Sphere(10, np.array(desired_isocenter))
            ray_mesh += pv.Line(
                np.array(desired_isocenter),
                np.array(desired_isocenter + ray_world * 400),
            )
            self.show_scene_cadaver(
                ray_mesh, patient_array_from_loopx=self.acquisition_1.patient_array_from_loopx
            )

        self.first_image_info = dict(
            dicom_path=dicom_path,
            image_paths=image_paths,
            image=image,
            kwire_mask=kwire_mask,
            hip_mask=hip_mask,
            startpoint_heatmap=startpoint_heatmap,
            startpoint_depthmap=startpoint_depthmap,
            corridor_mask=corridor_mask,
            startpoint_in_index=startpoint_in_index,
            direction_in_index=direction_in_index,
            corridor_line_in_index=corridor_line_in_index,
            index_from_patient_array=index_from_patient_array,
            startpoint_z=z_cam,
            desired_isocenter=desired_isocenter,
            ray_world=ray_world,
            device=device_config,
            image_time=image_timestamp,
            loopx_pose=loopx_pose,
            loopx_pose_planned=self.device.get_pose(),
        )
        log.info("Saving first image info.")
        # utils.save_json(self.images_dir / f"{base}_info.json", self.first_image_info)
        log.info("Done.")
        self.first_image_success = True
        return dict(success=True, **self.device.get_pose())

    def run_second_image(
        self,
        loopx_pose: Dict[str, float],
        image_timestamp: str,
    ):
        """Run the algorithm on the second image (for a cadaver study)."""
        log.info("Second image of the cadaver.")

        if not self.first_image_success or self.acquisition_1 is None:
            log.warning("First image was not successful.")
            return dict(success=False)

        meshes = []
        if image_timestamp == "":
            if self.ct is None or self.projector is None:
                log.warning("No CT provided to use for DRR second image.")
                return
            log.info("Analyzing DRR for second image.")
            loopx_pose = self.first_image_info["loopx_pose_planned"]
            dicom_path = None
            device_time = datetime.now()
            patient_array_from_loopx = self.device.world_from_loopx
            ray_world_2 = self.device.principle_ray_in_world
            image = self.projector()
            _image = image.copy()
            index_from_patient_array_2 = self.device.index_from_world
        else:
            dicom_path, device_time = self.get_closest_image(image_timestamp)
            if dicom_path is None or not dicom_path.exists():
                log.warning("No image found.")
                return

            self.acquisition_2 = Acquisition(dicom_path)
            patient_array_from_loopx = self.acquisition_2.patient_array_from_loopx
            ray_world_2 = self.acquisition_2.principle_ray_in_world
            image = self.acquisition_2.image
            _image = deepdrr.utils.neglog(image)
            self.device.sensor_height = image.shape[0]
            self.device.sensor_width = image.shape[1]
            index_from_patient_array_2 = self.acquisition_2.index_from_patient_array

        # Reposition virtual device.
        self.device.reposition(patient_array_from_loopx, **loopx_pose)
        ray_world_1 = self.first_image_info["ray_world"]

        # DRR for verification
        if self.projector is not None and self.acquisition_2 is not None:
            drr = self.projector(self.acquisition_2.index_from_patient_array)
        else:
            drr = None

        # Analyze the image.
        (
            kwire_mask,
            hip_mask,
            startpoint_heatmap,
            startpoint_depthmap,
            corridor_mask,
        ) = self.analyze_image(_image)
        (
            corridor_line_in_index_2,
            startpoint_in_index_2,
            direction_in_index_2,
        ) = eval_utils.locate_corridor(startpoint_heatmap, corridor_mask, mask=hip_mask)
        if (
            corridor_line_in_index_2 is None
            or startpoint_in_index_2 is None
            or direction_in_index_2 is None
        ):
            log.warning("Could not locate corridor.")
            return dict(success=False)

        # Get the values from the first image
        startpoint_in_index_1 = self.first_image_info["startpoint_in_index"]
        index_from_patient_array_1: geo.CameraProjection = self.first_image_info[
            "index_from_patient_array"
        ]
        corridor_line_in_index_1: geo.Line2D = self.first_image_info["corridor_line_in_index"]
        startpoint_in_index_1: geo.Point2D = self.first_image_info["startpoint_in_index"]
        direction_in_index_1: geo.Vector2D = self.first_image_info["direction_in_index"]

        # Get the corridor in world space as the intersection of planes
        corridor_plane_1 = corridor_line_in_index_1.backproject(index_from_patient_array_1)
        corridor_plane_2 = corridor_line_in_index_2.backproject(index_from_patient_array_2)
        corridor_in_patient_array = corridor_plane_1.meet(corridor_plane_2)

        # Get the approximate startpoint by intersecting the ray from the first view with the second plane.
        startpoint_ray_1 = startpoint_in_index_1.backproject(index_from_patient_array_1)
        approx_startpoint_in_patient_array = startpoint_ray_1.meet(corridor_plane_2)
        startpoint_in_patient_array = corridor_in_patient_array.project(
            approx_startpoint_in_patient_array
        )
        direction_in_patient_array = corridor_in_patient_array.get_direction()

        #  ensure pointed in the right direction
        projected_direction = (
            index_from_patient_array_2 @ (startpoint_in_patient_array + direction_in_patient_array)
            - startpoint_in_index_2
        )
        if projected_direction.dot(direction_in_index_2) < 0:
            direction_in_patient_array = -direction_in_patient_array

        if self.adjust_corridor:
            log.info(f"Adjusting corridor...")
            (
                startpoint_in_patient_array,
                direction_in_patient_array,
                radius,
            ) = eval_utils.adjust_corridor(
                corridor_heatmap_1=self.first_image_info["corridor_mask"],
                corridor_heatmap_2=corridor_mask,
                hip_mask_1=self.first_image_info["hip_mask"],
                hip_mask_2=hip_mask,
                index_from_world_1=index_from_patient_array_1,
                index_from_world_2=index_from_patient_array_2,
                startpoint_in_world=startpoint_in_patient_array,
                direction_in_world=direction_in_patient_array,
                image_1=self.first_image_info["image"],
                image_2=image,
                radius=self.adjustment_radius,
            )
            corridor_in_patient_array = geo.line(
                startpoint_in_patient_array, direction_in_patient_array
            )
            log.info(f"Done adjusting corridor.")

        self.patient_array_from_corridor = geo.FrameTransform.from_pointdir(
            startpoint_in_patient_array, direction_in_patient_array
        )
        self.corridor_radius = radius
        self.publish_corridor()

        # Check for K-wires in both images and, if present, triangulate and send to hololens.
        # Remember, the k-wire direction points from the tip to the back of the K-wire.
        kwire_mask_1 = self.first_image_info["kwire_mask"]
        (
            kwire_line_in_index_1,
            kwire_tip_in_index_1,
            kwire_direction_in_index_1,
        ) = eval_utils.locate_corridor(
            startpoint_heatmap=None,
            corridor_heatmap=kwire_mask_1,
            startpoint=startpoint_in_index_1,
            move_startpoint_to_mask=True,
            threshold=0.5,
        )
        (
            kwire_line_in_index_2,
            kwire_tip_in_index_2,
            kwire_direction_in_index_2,
        ) = eval_utils.locate_corridor(
            startpoint_heatmap=None,
            corridor_heatmap=kwire_mask,
            startpoint=startpoint_in_index_2,
            move_startpoint_to_mask=True,
            threshold=0.5,
        )
        if (
            kwire_line_in_index_1 is None
            or kwire_tip_in_index_1 is None
            or kwire_direction_in_index_1 is None
            or kwire_line_in_index_2 is None
            or kwire_tip_in_index_2 is None
            or kwire_direction_in_index_2 is None
        ):
            log.info(f"Could not locate K-wire in both images.")
            kwire_tip_in_patient_array = None
            kwire_direction_in_patient_array = None
        else:
            kwire_plane_1 = kwire_line_in_index_1.backproject(index_from_patient_array_1)
            kwire_plane_2 = kwire_line_in_index_2.backproject(index_from_patient_array_2)
            kwire_in_patient_array = kwire_plane_1.meet(kwire_plane_2)

            # Get the tip location
            kwire_tip_ray_1 = self.acquisition_1.backproject(kwire_tip_in_index_1)
            approx_kwire_tip_in_patient_array = kwire_tip_ray_1.meet(kwire_plane_2)
            kwire_tip_in_patient_array = kwire_in_patient_array.project(
                approx_kwire_tip_in_patient_array
            )
            kwire_direction_in_patient_array = kwire_in_patient_array.get_direction()

            # ensure pointed in the right direction
            kwire_projected_direction = (
                index_from_patient_array_2
                @ (kwire_tip_in_patient_array + kwire_direction_in_patient_array)
                - kwire_tip_in_index_2
            )
            # TODO: should be < 0, but for some reason it's > 0
            if kwire_projected_direction.dot(kwire_direction_in_index_2) > 0:
                kwire_direction_in_patient_array = -kwire_direction_in_patient_array

            self.patient_array_from_kwire = geo.FrameTransform.from_pointdir(
                kwire_tip_in_patient_array, kwire_direction_in_patient_array
            )
            self.publish_kwire()

        # Backproject for sanity check.
        image_1 = self.first_image_info["image"]
        image_2 = image
        corridor_image_1 = image_utils.draw_line(
            image_1,
            corridor_line_in_index_1,
        )
        corridor_image_2 = image_utils.draw_line(
            image_2,
            corridor_line_in_index_2,
        )

        projected_corridor_image_1 = image_utils.draw_line(
            image_1,
            self.acquisition_1.index_from_patient_array @ corridor_in_patient_array,
        )
        if self.acquisition_2 is None:
            projected_corridor_image_2 = image_utils.draw_line(
                image_2,
                index_from_patient_array_2 @ corridor_in_patient_array,
            )
        else:
            projected_corridor_image_2 = image_utils.draw_line(
                image_2,
                self.acquisition_2.index_from_patient_array @ corridor_in_patient_array,
            )

        base = "2nd"
        log.info(f"Saving images...")
        self.save_images(
            base,
            image=image,
            # kwire_mask=combine_heatmap(image, kwire_mask > 0.5),
            # hip_mask=combine_heatmap(image, hip_mask > 0.5),
            startpoint_heatmap=combine_heatmap(image, startpoint_heatmap),
            # startpoint_depthmap=combine_heatmap(image, startpoint_depthmap),
            corridor_mask=combine_heatmap(image, corridor_mask),
            corridor_image_1=corridor_image_1,
            corridor_image_2=corridor_image_2,
            projected_corridor_1=projected_corridor_image_1,
            projected_corridor_2=projected_corridor_image_2,
        )
        log.info(f"Done saving images.")

        meshes = []
        corridor_mesh = pv.Sphere(10, np.array(startpoint_in_patient_array))
        corridor_mesh += pv.Line(
            np.array(startpoint_in_patient_array - direction_in_patient_array * 400),
            np.array(startpoint_in_patient_array + direction_in_patient_array * 400),
        )
        meshes.append(corridor_mesh)

        if kwire_tip_in_patient_array is not None and kwire_direction_in_patient_array is not None:
            kwire_mesh = pv.Sphere(10, np.array(kwire_tip_in_patient_array))
            kwire_mesh += pv.Line(
                np.array(kwire_tip_in_patient_array - kwire_direction_in_patient_array * 400),
                np.array(kwire_tip_in_patient_array + kwire_direction_in_patient_array * 400),
            )
            meshes.append(kwire_mesh)

        # Print the deviations
        if kwire_tip_in_patient_array is not None and kwire_direction_in_patient_array is not None:
            kwire_to_corridor_angle = kwire_in_patient_array.angle(corridor_in_patient_array)
            kwire_to_startpoint_distance = kwire_in_patient_array.distance(
                startpoint_in_patient_array
            )
            log.info(
                f"\n\nK-Wire to corridor error: {kwire_to_startpoint_distance:5.1f} mm, {math.degrees(kwire_to_corridor_angle):5.2f}{DEGREE_SIGN}\n\n"
            )

        self.show_scene_cadaver(*meshes, patient_array_from_loopx=patient_array_from_loopx)
