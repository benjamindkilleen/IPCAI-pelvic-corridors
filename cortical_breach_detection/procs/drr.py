import math
import logging
from pathlib import Path
from termios import TAB0
import traceback
from typing import Any, Dict, Union, List, Tuple, Optional
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import time
import datetime
import multiprocessing as mp
from scipy.spatial.transform import Rotation

import deepdrr
from deepdrr import geo
from deepdrr.utils import data_utils, image_utils
import zmq
import threading

from ..loopx import LoopX
from ..threads import ImagePublisher, PosePublisher

log = logging.getLogger(__name__)


# TODO: make process runnable, currently getting pickle errors

flipy = np.ones((4, 4))
flipy[1, 0] = -1
flipy[0, 1] = -1
flipy[2, 1] = -1
flipy[1, 2] = -1
flipy[1, 3] = -1
flipz = np.ones((4, 4))
flipz[2, 0] = -1
flipz[2, 1] = -1
flipz[0, 2] = -1
flipz[1, 2] = -1
flipz[2, 3] = -1


class DRREngine(mp.Process):

    # There are things that need to be communicated to this process:
    # 1. The pose of the handle, which gives the desired viewing direction.
    # 2. The pose of the CT in the world.
    # 3. The pose of the Loop-X gantry in the world, when the Loop-X is in a known position.

    # The queue contains updates these transforms. "World" is the deepdrr world, in mm. This has the same origin as the HoloLens world (which is in m).
    # - world_from_handle
    # - world_from_ct for the CT anatomical coordinates
    # - world_from_reference for the BrainLab reference marker.
    # - reference_from_loop_x_gantry, as provided by a thread that periodically checks if the tracking info has been updated.

    # Be sure to use a queue for each of these.

    # The main loop of the process should follow these steps:
    # 1. Check if the shared handle pose has been updated. If it has, update the a local variable of the desired view.
    # 2. Check if the shared CT pose has been updated. If it has, update the Volume transform to match.
    # 3. Check if the shared Loop-X pose has been updated. If it has, update the Loop-X transform to match.
    # 4. Take a DRR image.
    # 5. Transmit the DRR image to the HoloLens.

    # Contains the 4x4 transformation matrix giving the pose of the handle, which has its origin at
    # the "center" and Z-axis in the direction of the principle ray.

    # TODO: the rest of the communicated variables, in queues.

    # NOTE: pelvis model is in RAS coordinates. This is the RAS transform, derived from LPS_from_RAS @ model_from_marker @ RAS_from_LPS.
    pelvis_model_from_pelvis_marker = geo.frame_transform(
        """-0.91108588 -0.39857756 -0.13363132 25.74871523
        -0.21853024 0.17657402 0.96325751 -43.43864193
        -0.35911828 0.90374578 -0.24713664 -153.38330693
        0.00000000 0.00000000 0.00000000 1.00000000"""
    )
    # This is the one you want to provide to Slicer

    # The original is:
    """-0.91108588 -0.39857756 0.13363132 -25.74871523                                                                                                        
                    -0.21853024 0.17657402 -0.96325751 43.43864193                                                                                                         
                    0.35911828 -0.90374578 -0.24713664 -153.38330693                                                                                                       
                    0.00000000 0.00000000 0.00000000 1.00000000 """

    pelvis_marker_from_pelvis_model = pelvis_model_from_pelvis_marker.inv
    # f = geo.unity_from_slicer @ pelvis_marker_from_pelvis_model
    # print(f.data.reshape(-1))
    # stl_from_obj = geo.frame_transform(Rotation.from_euler("x", -90, degrees=True))

    # # Conversion from right-handed to left-handed coordinate system.
    # marker_from_model_rhs = (pelvis_marker_from_pelvis_model).data.copy()
    # marker_from_model_rhs[:3, 3] = marker_from_model_rhs[:3, 3] / 1000.0
    # marker_from_model_lhs = marker_from_model_rhs * flipz
    # q = Rotation.from_matrix(marker_from_model_lhs[:3, :3]).as_quat()
    # zxy = Rotation.from_matrix(marker_from_model_lhs[:3, :3]).as_euler(
    #     "zxy", degrees=True
    # )
    # xyz = zxy[[1, 2, 0]]
    # t = marker_from_model_lhs[:3, 3]

    # print(
    #     f"TODO: fix this. pelvis_marker_from_pelvis_model:\nq: {q}\nxyz: {xyz}\nt: {t}"
    # )

    # loopx is the loopx marker frame
    loopx_from_badgantry = geo.frame_transform(
        """-0.0944683	0.232537	0.967989	971.437
        0.995272	0	0.0971309	97.4769
        0.0225865	0.972588	-0.231438	-232.262
        0	0	0	1"""
    )

    # From 3-point calibration
    #     badgantry_from_gantry = geo.frame_transform(
    #         """-0.98268527 0.15171893 0.10635313 0.07345377
    # 0.13418525 0.97857666 -0.15614745 -0.09838375
    # -0.12776522 -0.13917278 -0.98199135 0.02358579
    # 0.00000000 0.00000000 0.00000000 1.00000000"""
    #     )

    # From PnP calibration in the AP
    badgantry_from_gantry = geo.frame_transform(
        """ -0.95403904 0.29096690 0.07174767 -0.00000003
0.27102152 0.93987828 -0.20778878 -0.00000004
-0.12789373 -0.17879346 -0.97553885 -0.00000001
0.00000000 0.00000000 0.00000000 1.00000000"""
    )
    loopx_from_loopx_gantry = loopx_from_badgantry @ badgantry_from_gantry

    pointer_from_back = geo.frame_transform(
        """1	0	0	-11.7029
        0	-1	-1.22465e-16	9.92828
        0	1.22465e-16	-1	-83.6439
        0	0	0	1"""
    )
    pointer_from_tip = geo.frame_transform(
        """0.836488	0	0.547985	140.63
        -0.0836032	0.988293	0.127619	32.7509
        -0.54157	-0.152565	0.826696	212.156
        0	0	0	1"""
    )

    pointer_from_handle = geo.FrameTransform.from_line_segments(
        pointer_from_tip.o,
        pointer_from_tip.o + (pointer_from_back.o - pointer_from_tip.o).hat(),
        geo.p(0, 0, 0),
        geo.p(0, 0, 1),
    )

    handle_types = ["AR Handle", "Pointer"]

    def __init__(
        self,
        queue: mp.Queue,
        loopx_queue: mp.Queue,
        publish_queue: mp.Queue,
        pose_queue: mp.Queue,
        data_dir: str,
        port: int = 12350,
        fps: int = 12,
        pose_port: int = 12351,
        handle_type: str = "AR Handle",
        device: Dict[str, Any] = {},
        projector: Dict[str, Any] = {},
    ):
        """
        Initialize the DRR engine.

        Args:
            queue: The queue to use to receive updates to the transforms and other commands.
            loopx_queue: The queue to use to send updates to the Loop-X pose.
            data_dir: The directory where the CT files are found.
            port: The port to use to publish images.
            fps: The maximum frames per second to use.
            pose_port: The port to use to publish loopx component transforms.
        """
        super(DRREngine, self).__init__()
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            self.data_dir.mkdir()

        self.queue = queue
        self.loopx_queue = loopx_queue
        self.image_queue = publish_queue
        self.pose_queue = pose_queue
        self.port = port
        self.fps = fps
        self.seconds_per_image = 1.0 / fps
        self.pose_port = pose_port

        log.info(
            f"pointer from_handle: {geo.Rotation.from_matrix(self.pointer_from_handle.R).as_euler('xyz')}"
        )

        self.image_dir = Path("images")
        self.image_dir.mkdir()
        self.info_dir = Path("info")
        self.info_dir.mkdir()
        self.device_config = device
        self.projector_config = projector

        self.ct_path = self.data_dir / "2 LFOV_FullScan (2022-08-17 2239).nii.gz"

        self.download()

        # Local variables for different poses.
        self.world_from_handle = geo.FrameTransform.identity()
        self.world_from_reference = geo.FrameTransform.identity()
        self.reference_from_loopx_gantry = geo.frame_transform(
            """
            0.27914 0.646374 -0.710128 116.547 
-0.898622 0.436505 0.0440829 -347.013 
0.338468 0.625831 0.702692 -487.055 
0 0 0 1 

            """
        )
        self.loopx_pose = [180, 360, 0, 0, 0, 0]

        # Used to indicate the users are currently imaging, so no updates to the DRR should be made.
        self.imaging = False
        self.handle_type = handle_type
        self.running = False
        self.image_publisher = None
        self.projector = None

    def run(self):
        self.image_publisher = ImagePublisher(self.image_queue, self.port)
        self.image_publisher.start()

        self.pose_publisher = PosePublisher(self.pose_queue, self.pose_port)
        self.pose_publisher.start()

        self.device = LoopX(**self.device_config)
        self.ct = deepdrr.Volume.from_nifti(self.ct_path, use_cached=True)
        self.projector = deepdrr.Projector(self.ct, carm=self.device, **self.projector_config)
        self.projector.initialize()

        self.running = True
        self.imaging = False
        update_drr = False

        t0 = time.time()
        t_drr = time.time()
        while self.running:
            try:
                if (
                    not self.imaging
                    and update_drr
                    and time.time() - t_drr >= self.seconds_per_image
                ):
                    # log.info(f"DRR Engine: Updating DRR")
                    self.image = self.projector()
                    self.image_info = dict(
                        device=self.device.get_config(),
                        ct=self.ct.get_config(),
                    )
                    self.publish_image()
                    update_drr = False
                    t_drr = time.time()

                # Check the queue for new info
                item = self.queue.get()  # blocking
                if item is None:
                    # This is redundant, because the above call is blocking.
                    if time.time() - t0 > 5:
                        log.info("DRR Engine waiting...")
                        t0 = time.time()
                    time.sleep(0.1)
                    continue

                message_type = item[0]
                update_device = False

                if message_type == "world_from_handle" and self.handle_type == "AR Handle":
                    # Obtained from HoloLens virtual object (or tracked tool)
                    self.world_from_handle.data[:] = item[1]
                    update_device = True
                elif message_type == "world_from_handle":
                    # log.info(f"Ignoring AR Handle update")
                    pass
                elif message_type == "world_from_pointer" and self.handle_type == "Pointer":
                    world_from_pointer = geo.FrameTransform(item[1])
                    self.world_from_handle = world_from_pointer @ self.pointer_from_handle
                    update_device = True
                elif message_type == "world_from_pointer":
                    # log.info(f"Ignoring Pointer update")
                    pass
                elif message_type == "world_from_reference":
                    self.world_from_reference.data[:] = item[1]
                elif message_type == "world_from_pelvis_marker":
                    # Obtained from HoloLens tracking of CT marker
                    world_from_pelvis_marker = geo.frame_transform(item[1])
                    world_from_anatomical = (
                        world_from_pelvis_marker
                        @ self.pelvis_marker_from_pelvis_model
                        @ geo.LPS_from_RAS
                    )
                    self.ct.world_from_anatomical.data[:] = world_from_anatomical.data
                    update_drr = True
                elif message_type == "reposition_loopx":
                    # User entered current pose in the GUI, so reposition the loop-x.
                    log.info(f"Repositioning Loop-X...")
                    for i, x in enumerate(item[1]):
                        if x is not None:
                            self.loopx_pose[i] = x
                    if item[2] is not None:
                        self.reference_from_loopx = geo.frame_transform(item[2])
                    world_from_loopx = self.world_from_reference @ self.reference_from_loopx
                    self.device.reposition(world_from_loopx, *self.loopx_pose)
                    log.info(f"Repositioned Loop-X (check markers and source/detector boxes)")
                    self.publish_actual_device_pose()
                    update_device = True
                elif message_type == "shoot":
                    # Most recent (already viewed) image is ready to be saved.
                    self.imaging = True
                    self.save_image()
                    self.loopx_queue.put(self.device.get_pose())
                elif message_type == "imaging_done":
                    self.imaging = False
                elif message_type == "handle_type":
                    if item[1] in self.handle_types:
                        self.handle_type = item[1]
                        update_device = True
                    else:
                        log.warning(f"Unknown handle type: {item[1]}")
                else:
                    log.warning(f"Unknown item in queue: {item}")

                if not self.imaging and update_device:
                    # Move the virtual Loop-X to the *desired* position.
                    # log.info("DRR Engine: Updating device")
                    # Reset the pose to the real device, so that this is always the initialization.
                    self.device.align(
                        self.world_from_handle @ geo.p(0, 0, 0),
                        self.world_from_handle @ geo.v(0, 0, 1),
                    )
                    self.publish_device_pose()

                    # log.info(f"Loop-X:\n{self.device}")

                    # log.info(f"device:\n{self.device}")
                    update_device = False
                    update_drr = True

            except KeyboardInterrupt:
                break
            except Exception as e:
                traceback.print_exc()
                continue

    def publish_actual_device_pose(self):
        """Only call when self.device in `actual` pose corresponding to real loop-x pose."""
        source_pose = self.device.world_from_source.data
        detector_pose = self.device.world_from_detector.data
        gantry_pose = self.device.world_from_gantry.data
        self.pose_queue.put((3, source_pose))
        self.pose_queue.put((4, detector_pose))
        self.pose_queue.put((5, gantry_pose))

    def publish_device_pose(self):
        source_pose = self.device.world_from_source.data
        detector_pose = self.device.world_from_detector.data

        # order corresponds to order of game objects in unity
        self.pose_queue.put((0, source_pose))
        self.pose_queue.put((1, detector_pose))

        world_from_loopx = (
            self.world_from_reference
            @ self.reference_from_loopx_gantry
            @ self.loopx_from_loopx_gantry.inv
        )
        self.pose_queue.put((2, world_from_loopx.data))

    def publish_image(self):
        """
        Send the current image to the HoloLens.

        Args:
            image: The image to send, as a float32 numpy array in [0,1].
        """
        image = self.image
        image = (image * 255).astype(np.uint8)
        self.image_queue.put(image)

    def save_image(self):
        """
        Save an image and info to the disk.
        """
        image = self.image
        image_info = self.image_info

        image = (image * 255).astype(np.uint8)

        # Save the image.
        now = datetime.datetime.now()
        stem = f"{now:%Y-%m-%d_%H-%M-%S}"
        image_path = self.image_dir / f"{stem}_drr.png"
        info_path = self.info_dir / f"{stem}_info.json"
        Image.fromarray(image).save(image_path)
        data_utils.save_json(info_path, image_info)
        log.info(f"Saved image to {image_path}")

    def download(self):
        if not self.data_dir.exists():
            self.data_dir.mkdir()

        cache_dir = self.data_dir / "cache"
        if not cache_dir.exists():
            cache_dir.mkdir()

        data_utils.download(
            url="https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/bkillee1_jh_edu/ETigA_TIhIFNq__0UASfV1cB6KsDhMOKH24QeQWdwgIKgQ?e=EbIZo7&download=1",
            filename=self.ct_path.name,
            root=self.data_dir,
        )

        data_utils.download(
            url="https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/bkillee1_jh_edu/ER5iLnmprhxGiT9w_umskugBof-N7-uOsEJHoLGqIYtdmA?e=wHEe2x&download=1",
            filename=self.ct_path.name,
            root=cache_dir,
        )

        stem = self.ct_path.stem.split(".")[0]
        data_utils.save_json(cache_dir / f"{stem}.json", dict(bone=1, air=0))

    def close(self):
        self.running = False
        if self.projector is not None:
            self.projector.free()
        if self.image_publisher is not None:
            self.image_publisher.close()
