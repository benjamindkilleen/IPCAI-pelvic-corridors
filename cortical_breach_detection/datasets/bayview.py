from __future__ import absolute_import
from __future__ import annotations

import deepdrr
import logging
import random
import re
from copy import deepcopy
from deepdrr import geo
from deepdrr import vis
from deepdrr.annotations.line_annotation import LineAnnotation
from deepdrr.device import MobileCArm
from deepdrr.utils import data_utils
from deepdrr.utils import image_utils
from pathlib import Path
from shutil import rmtree
from time import sleep
from typing import Any
from typing import Dict
from typing import Generator
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Union

import h5py
import numpy as np
import pytorch_lightning as pl
import pyvista as pv
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from rich.progress import Progress
from scipy.spatial import KDTree
from skimage.exposure import match_histograms
from torch.utils.data import DataLoader

from .. import tools
from .. import utils
from ..console import console
from .ctpelvic1k import CTPelvic1K
from .ctpelvic1k import get_kwire_guide

log = logging.getLogger(__name__)


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
    tmp1 = (
        ITK_xform[0, 0] * ITK_xform[0, 3]
        + ITK_xform[0, 1] * ITK_xform[1, 3]
        + ITK_xform[0, 2] * ITK_xform[2, 3]
    )
    tmp2 = (
        ITK_xform[1, 0] * ITK_xform[0, 3]
        + ITK_xform[1, 1] * ITK_xform[1, 3]
        + ITK_xform[1, 2] * ITK_xform[2, 3]
    )
    tmp3 = (
        ITK_xform[2, 0] * ITK_xform[0, 3]
        + ITK_xform[2, 1] * ITK_xform[1, 3]
        + ITK_xform[2, 2] * ITK_xform[2, 3]
    )

    ITK_xform[0, 3] = -tmp1
    ITK_xform[1, 3] = -tmp2
    ITK_xform[2, 3] = -tmp3
    return ITK_xform


def read_itk_affine_transform_from_file(path: str) -> geo.FrameTransform:
    f = h5py.File(path)
    transform_group = f["TransformGroup"]["0"]
    itk_xform = np.array(transform_group["TranformParameters"])
    itk_xform = itk_xform.reshape(4, 3).T
    # TODO: not sure this is the right way to load these.
    return geo.FrameTransform.from_rt(rotation=itk_xform[:, :3].T, translation=itk_xform[:, 3])


def read_triangulated_fiducials(path: str) -> Tuple[geo.Point3D, geo.Point3D]:
    with open(path) as file:
        lines = file.readlines()

    last_ring_center = geo.point(*map(float, lines[3].split(",")[1:4]))
    kwire_tip = geo.point(*map(float, lines[4].split(",")[1:4]))

    if int(lines[1].split("=")[-1].strip()) == 0:
        # convert to RAS
        last_ring_center = geo.RAS_from_LPS @ last_ring_center
        kwire_tip = geo.RAS_from_LPS @ kwire_tip

    return last_ring_center, kwire_tip


class Bayview:

    # Download command:
    # wget -O 2022-02_Bayview_Cadaver.zip https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/bkillee1_jh_edu/EcHlmhUutalPvIimJ3SGXXsBEMlfqByS8Jie-R1hNSdclQ\?e\=sqGquk\&download\=1

    # Which images should be used in the test set
    image_ids = [
        38,
        41,
        # 44,
        45,  # bad
        50,  # bad
        52,  # bad
        56,
        60,  # bad
        62,
        69,
    ]

    def __init__(
        self,
        root: str = "~/datasets",
        mode: Literal["train", "val", "test"] = "test",
        download: bool = False,
        image_size: Union[int, Tuple[int, int]] = None,
        kwire_guide: Optional[str] = None,
        kwire_guide_density: float = 0.05,
        corridor_depth: bool = True,
        test_only: bool = True,
    ):
        """Make the dataset.

        Args:
            root (str, optional): Where datasets are stored in general. Defaults to "~/datasets".
            download: Download the CT or X-Ray data.

        """
        self.root = Path(root).expanduser()
        self.dataset_dir = self.root / "2022-02_Bayview_Cadaver"
        self.mode = mode
        self.image_size = deepdrr.utils.tuplify(image_size, 2)
        self.kwire_guide = kwire_guide
        self.kwire_guide_density = kwire_guide_density
        self.corridor_depth = corridor_depth
        self.test_only = test_only

        self.ct_dir = self.dataset_dir / "2022-02-22_Cadaver_Femur"
        self.ct_path = self.ct_dir / "Spec22-2181-CT-Bone-1mm.nii.gz"
        self.seg_path = self.ct_dir / "Spec22-2181-Seg-Bone-1mm.nii.gz"
        self.gt_trajectories_dir = self.dataset_dir / "2022-02-22_Cadaver_Femur_trajectories"
        self.gt_trajectory_path = (
            self.gt_trajectories_dir / "2022-02-22_Cadaver_Femur_right_kwire_trajectory.mrk.json"
        )
        self.trajectory_dir = self.dataset_dir / "trajectories"

        self.surface_data_dir = self.dataset_dir / "surfaces"  # created by dataset
        self.surface_path = self.surface_data_dir / "2022-02-22_Cadaver_Femur_surface"

        self.original_images_dir = self.dataset_dir / "BayviewExp_Feb25" / "Xraytiff_ori"
        self.images_dir = self.dataset_dir / "processed_images"

        handeyeX_path = self.dataset_dir / "KwireExp_Benjamin" / "handeye_result" / "handeye_X.h5"
        self.handeyeX_transform = read_itk_affine_transform_from_file(handeyeX_path)

        # Cong's original
        # self.intrinsic = geo.CameraIntrinsicTransform(np.array([[-5257.73, 0, 767.5], [0, -5257.73, 767.5], [0, 0, 1]]))

        # I think my convention
        self.intrinsic = geo.CameraIntrinsicTransform(
            np.array(
                [[-1, 0, 1536], [0, -1, 1536], [0, 0, 1]]
            )  # Weird conversion I don't understand
            @ np.array([[-5257.73, 0, 767.5], [0, -5257.73, 767.5], [0, 0, 1]])
        )

        # Called "extrinsic" in rob's code. Not really the extrinsic matrix, just something Rob does.
        # See https://github.com/rg2/xreg/blob/ae2ab800bc2dad25c0785d138a4ccfe3136d2ad2/lib/file_formats/xregCIOSFusionDICOM.cpp
        self.extrinsic = geo.FrameTransform(
            np.array(
                [
                    [
                        0.0014173902529073,
                        0.0000057733016248,
                        0.9999990957527923,
                        -157.5822163123799555,
                    ],
                    [
                        -0.9999907608500266,
                        -0.0040758357952800,
                        0.0014174006256202,
                        -2.6757496083693013,
                    ],
                    [
                        0.0040758413782892,
                        -0.9999918324063819,
                        -0.0000000000000000,
                        -631.7764689585426368,
                    ],
                    [0, 0, 0, 1],
                ]
            )
        )

        self.takehome_dir = self.dataset_dir / "BayviewExp_Feb25" / "runs" / "run_takehome"
        if download:
            self.download()

        if self.mode == "test":
            self.image_paths = self.process_images()
        else:
            self.image_paths = []

    def _check_exists(self):
        raise NotImplementedError

    def download(self):
        if self._check_exists():
            return
        raise NotImplementedError

    def _get_image_info(self, image_path: Path):
        image_path = Path(image_path)
        return utils.load_json(image_path.parent / f"{image_path.stem}_info.json")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        image_id = image_path.stem

        # get the insertion depth for metrics
        image_info = self._get_image_info(image_path)

        # Get the triangulated fiducials in anatomical coordinates.
        fiducial_dir = self.takehome_dir / "output_triangulation"
        log.debug(f"image_id: {image_id}")
        fiducial_path = list(fiducial_dir.glob(f"triangulated_fidest{image_id}_*_*.fcsv"))[0]
        last_ring_center_in_anatomical, kwire_tip_in_anatomical = read_triangulated_fiducials(
            fiducial_path
        )

        transform_path = (
            self.takehome_dir / "output_mv" / f"pelvis_mv_regi_xform_fidest{image_id}.h5"
        )
        pelvis_regi_xform = read_itk_affine_transform_from_file(transform_path)
        # called pelvis_regis_xform in Cong's code, really the pelvis_from_carm

        # Let's have the world be the anatomical coordinates, because why not
        # Rob's camera
        camera3d_from_robcam = geo.FrameTransform(
            np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        )
        camera3d_from_world = camera3d_from_robcam @ self.extrinsic @ pelvis_regi_xform.inv
        # log.debug(f"pelvis xform:\n{pelvis_regi_xform}")
        # log.debug(f"pelvis xform inverse:\n{pelvis_regi_xform.inv}")
        # log.debug(f"extrinsic:\n{self.extrinsic}")
        # log.debug(f"intrinsic:\n{self.intrinsic}")
        # log.debug(f"camera3d from world:\n{camera3d_from_world}")

        # startpoint = last_ring_center_in_anatomical
        startpoint = geo.point(
            last_ring_center_in_anatomical
            + 20 * geo.vector(kwire_tip_in_anatomical - last_ring_center_in_anatomical).hat()
        )
        endpoint = kwire_tip_in_anatomical
        camera_projection = geo.CameraProjection(
            self.intrinsic, camera3d_from_robcam.inv @ camera3d_from_world
        )
        index_from_world = camera_projection.index_from_world

        # log.debug(f"projection of tip, should be (780, 405): {index_from_world @ kwire_tip_in_anatomical}")
        # log.debug(f"projection of tip, but really should be (755.6325 1130.2122)?: {index_from_world @ kwire_tip_in_anatomical}")
        # 1484
        # 1483
        # log.debug(f"projection of wire guide front: {index_from_world @ startpoit}")

        # get the image and cortical breach label
        image = TF.pil_to_tensor(Image.open(image_path)).to(torch.float32) / 255.0

        gt_startpoint = geo.point(image_info["gt_startpoint"])
        gt_endpoint = geo.point(image_info["gt_endpoint"])
        gt_startpoint_in_index = index_from_world @ gt_startpoint
        gt_endpoint_in_index = index_from_world @ gt_endpoint
        gt_startpoint_in_camera = camera3d_from_world @ gt_startpoint
        gt_endpoint_in_camera = camera3d_from_world @ gt_endpoint
        a_j = int(gt_startpoint_in_index[0])
        a_i = int(gt_startpoint_in_index[1])
        b_j = int(gt_endpoint_in_index[0])
        b_i = int(gt_endpoint_in_index[1])
        startpoint_is_in_image = (0 <= a_i < image.shape[1]) and (0 <= a_j < image.shape[2])

        # if not startpoint_in_image:
        #     log.debug("startpoint not in the image")

        # Get the heatmap of the startpoint.
        startpoint_heatmap = utils.nn_utils.heatmap(
            a_i, a_j, scale=image.shape[1] / 80, size=image.shape[1:3]
        )
        corridor_heatmap = utils.nn_utils.line_segment_heatmap(
            a_i, a_j, b_i, b_j, scale=image.shape[1] / 80, size=image.shape[1:3]
        )
        heatmaps = torch.stack([startpoint_heatmap, corridor_heatmap], axis=0)

        # log.debug(f"heatmap (min, max, mean): {startpoint_heatmap.min(), startpoint_heatmap.max(), startpoint_heatmap.mean()}")

        # Distance map is all -1 except at the point that matters
        depth_map = -100 * np.ones(image.shape[1:3], dtype=np.float32)
        d = gt_startpoint_in_camera[2]
        if startpoint_is_in_image:
            depth_map[a_i, a_j] = (d - CTPelvic1K.DEPTH_MEAN) / CTPelvic1K.DEPTH_STD
            # log.debug(f"depth: {depth_map[a_i, a_j]}")

        # Regress the depth along the corridor.
        if self.corridor_depth:
            length = (gt_endpoint - gt_startpoint).norm()
            # log.debug(f"num steps: {1 / (self.carm.pixel_size / length)}")
            for progress in np.arange(0, 1, 0.01):
                p = gt_startpoint.lerp(gt_endpoint, progress)
                # log.debug(f"p: {p}, {type(p)}")
                # log.debug(f"transform: {camera3d_from_world}")
                p_camera = camera3d_from_world @ p
                p_index = index_from_world @ p
                p_j = int(p_index[0])
                p_i = int(p_index[1])
                d = p_camera[2]
                # log.debug(f"depth at {progress:.01f}: {d:.01f}")
                if (0 <= p_i < image.shape[1]) and (0 <= p_j < image.shape[2]):
                    depth_map[p_i, p_j] = (d - CTPelvic1K.DEPTH_MEAN) / CTPelvic1K.DEPTH_STD
        # utils.imshow_save("depth.png", depth_map)
        # log.debug(f"num pixels on trajectory: {(depth_map > -100).sum()}")
        depth_map = torch.tensor(depth_map)

        # phi, for metrics purposes
        v = gt_endpoint_in_index - gt_startpoint_in_index
        phi = np.arctan2(v[0], v[1])

        # tau_map
        tau_map = -100 * np.ones(image.shape[1:3], dtype=np.float32)
        w = gt_endpoint_in_camera - gt_startpoint_in_camera
        tau = np.arccos(w[1] / w.norm())
        # log.debug(f"tau: {np.degrees(tau)} degrees")
        # if "tau" in image_info and not np.isclose(image_info["tau"], tau):
        # TODO: this is only commented out because the most recent sim is wrong, but it's fine.
        # raise RuntimeError("saved tau does not match: {:.02f}, {:.02f}".format(tau, image_info["tau"]))

        if startpoint_is_in_image:
            tau_map[a_i, a_j] = tau / np.pi
        # utils.imshow_save("tau.png", tau_map, interpolation="antialiased")
        tau_map = torch.tensor(tau_map)

        # disk segmentation
        # mask_path = image_path.parent / f"{image_path.stem}_kwire-guide-mask.png"
        # kwire_guide_mask = TF.pil_to_tensor(Image.open(mask_path))[0]
        kwire_guide_mask = torch.zeros_like(depth_map).to(torch.long)
        kwire_guide_seg = F.one_hot(kwire_guide_mask, num_classes=3)
        kwire_guide_seg = torch.permute(kwire_guide_seg, (2, 0, 1))

        original_image_size = image.shape
        if self.image_size[0] != image.shape[1] and self.image_size[1] != image.shape[2]:
            # Resize to desired resolution for model
            # Rather than re-compute everything, just resize the heatmaps when done.
            image = TF.resize(image, self.image_size)
            # raise NotImplementedError("make sure mask and other things resized")

        cortical_breach = image_info["cortical_breach"]
        cortical_breach_label = torch.tensor([cortical_breach], dtype=torch.int64)
        target = dict(
            heatmaps=heatmaps,
            depth_map=depth_map,
            tau_map=tau_map,
            segs=kwire_guide_seg,
            depth_map_valid=(depth_map > -100).to(torch.float32),
            cortical_breach_label=cortical_breach_label,
        )

        left_side = True  # should be false but whatever
        info = dict(
            left_side=torch.tensor(left_side, dtype=torch.long),
            image_path=str(image_path),
            view_name="random",
            view_label=0,
            progress=-0.5,  # torch.tensor(image_info["progress"], dtype=torch.float32),
            gt_front_disk=torch.tensor(np.array(startpoint), dtype=torch.float32),
            startpoint=torch.tensor(np.array(startpoint), dtype=torch.float32),
            endpoint=torch.tensor(np.array(endpoint), dtype=torch.float32),
            ct_path=image_info["ct"],
            base=self.get_base_from_ct_path(),
            camera3d_from_world=torch.tensor(geo.get_data(camera3d_from_world)),
            index_from_world=torch.tensor(geo.get_data(index_from_world)),
            world_from_index=torch.tensor(geo.get_data(index_from_world.inv)),
            gt_startpoint=torch.tensor(image_info["gt_startpoint"], dtype=torch.float32),
            gt_endpoint=torch.tensor(image_info["gt_endpoint"], dtype=torch.float32),
            d=torch.tensor(d, dtype=torch.float32),
            phi=torch.tensor(phi, dtype=torch.float32),
            tau=torch.tensor(tau, dtype=torch.float32),
            startpoint_is_in_image=torch.tensor(1, dtype=bool),
            kwire_guide_progress=torch.tensor(-0.1, dtype=torch.float32),
            source_to_detector_distance=torch.tensor(1020, dtype=torch.float32),
            pixel_size=torch.tensor(0.194, dtype=torch.float32),
            original_image_size=torch.tensor(original_image_size, dtype=torch.long),
        )

        return image, target, info

    @staticmethod
    def get_base_from_ct_path(ct_path: Union[Path, str] = None) -> Optional[str]:
        """Get a base stem to use for annotation or image paths, corresponding to the CT.

        Args:
            ct_path (Path): Posixpath to the ct.

        Returns:
            Optional[str]: Either the base string, or None if parsing fails.
        """
        return "Spec22-2181-CT-Bone_1mm"

    def get_ct(self) -> Tuple[deepdrr.Volume, deepdrr.Volume, str]:
        """Get the CT and segmentation."""
        ct = deepdrr.Volume.from_nifti(self.ct_path, use_thresholding=True)
        ct.faceup()

        seg = deepdrr.Volume.from_nifti(
            self.seg_path, segmentation=True, world_from_anatomical=ct.world_from_anatomical
        )
        # Just get the right hip
        seg.data[251:, :, :] = 0
        seg.data[seg.data != 1] = 0
        return ct, seg

    def process_images(self):
        """Generate PNGs that resemble those made by DeepDRR.

        Args:
            overwrite (bool, optional): Whether to overwrite existing data. Defaults to False.
        """
        # TODO: ensure startpoint is in the image.

        if not self.surface_data_dir.exists():
            self.surface_data_dir.mkdir()

        if self.images_dir.exists() and len(list(self.images_dir.glob("*.png"))) == len(
            self.image_ids
        ):
            return sorted(list(self.images_dir.glob("*.png")))
        elif self.images_dir.exists():
            rmtree(self.images_dir)
        self.images_dir.mkdir()

        ct, seg = self.get_ct()
        fractured = False
        base = self.get_base_from_ct_path()
        surface_path = self.surface_data_dir / f"{base}_LPS.stl"
        if surface_path.exists():
            surface = pv.read(surface_path)
            surface.transform(geo.RAS_from_LPS.get_data(), inplace=True)
        else:
            surface = seg.isosurface(label=1)
            surf: pv.PolyData = surface.copy()
            surf.transform(geo.LPS_from_RAS.get_data(), inplace=True)
            surf.save(surface_path)
        surface.transform(ct.world_from_anatomical.get_data(), inplace=True)
        tree = KDTree(surface.points.astype(np.double))
        annotation = deepdrr.annotations.LineAnnotation.from_markup(self.gt_trajectory_path, ct)

        image_paths = []
        for n, image_id in enumerate(self.image_ids):
            log.info(f"processing image {image_id}")
            original_image_path = self.original_images_dir / f"{image_id}.tiff"
            stem = f"{image_id}"
            image = np.array(Image.open(original_image_path))

            # Take care of the border
            border = 20  # 10
            image_max = image[
                border : image.shape[0] - 2 * border, border : image.shape[1] - border
            ].max()
            image[:border] = image_max
            image[image.shape[0] - 2 * border :] = image_max
            image[:, :border] = image_max
            image[:, image.shape[1] - border :] = image_max

            # Do some stuff to make the needle lighter.
            # 10935 is the mean of the needle region, 15805 is roughly the max, maybe 17000 is a good dividing line
            # Other option is to use this to get rid of the needle in the mask
            # where_needle = image < 17000
            # label, num_features = ndimage.label()
            # image[image < 18000] += 7000

            image = deepdrr.utils.neglog(image)

            reference_image_path = Path(
                "~/projects/cortical-breach-detection/images/drr.png"
            ).expanduser()
            reference_image = np.array(Image.open(reference_image_path)).astype(np.float32) / 255.0
            image[
                border : image.shape[0] - 2 * border, border : image.shape[1] - border
            ] = match_histograms(
                image[border : image.shape[0] - 2 * border, border : image.shape[1] - border],
                reference_image[
                    border : image.shape[0] - 2 * border, border : image.shape[1] - border
                ],
            )
            image = np.clip(image, 0, 1)

            image_utils.save(self.images_dir / f"{stem}.png", image)

            # TODO: all of this based on Cong's triangulation and making the cortical breach detection a staticmethod
            # trajectory_annotation, cortical_breach, reason, save_trajectory = self.get_trajectory(
            #     trajectory_path=None if not trajectory_paths else trajectory_paths[image_index],
            #     annotation=annotation,
            #     tree=tree,
            #     ct=ct,
            #     seg=seg,
            #     surface=surface,
            #     fractured=fractured,
            # )
            # log.info(f"[{base}] cortical_breach: {cortical_breach}")
            # startpoint = trajectory_annotation.startpoint_in_world
            # endpoint = trajectory_annotation.endpoint_in_world
            # kwire.align(startpoint, endpoint, progress=progress)
            # length = (endpoint - startpoint).norm()
            # kwire_guide_progress = -np.random.uniform(15, 20) / length
            log.warn("pose, breach is faked")
            cortical_breach = True

            # camera3d_from_world = self.carm.camera3d_from_world
            # gt_startpoint = annotation.startpoint_in_world
            # gt_endpoint = annotation.endpoint_in_world
            # gt_startpoint_in_camera = camera3d_from_world @ gt_startpoint
            # gt_endpoint_in_camera = camera3d_from_world @ gt_endpoint
            # w = gt_endpoint_in_camera - gt_startpoint_in_camera
            # tau = np.arccos(w[1] / w.norm())
            # taus.append(tau)
            # # log.debug(f"tau: {np.degrees(tau):.01f} degrees")

            # Get some other nice info to have.
            # index_from_world = self.carm.get_camera_projection().index_from_world
            # principle_ray = self.carm.principle_ray_in_world
            info = dict(
                ct=self.ct_path,
                anatomical_from_world=ct.anatomical_from_world,
                anatomical_coordinate_system=ct.anatomical_coordinate_system,
                annotation=self.gt_trajectory_path.name,
                annotation_path=self.gt_trajectory_path,
                base=base,
                gt_startpoint=annotation.startpoint,  # anatomical is world
                gt_endpoint=annotation.endpoint,  # anatomical is world
                # startpoint=startpoint,
                # endpoint=endpoint,
                # startpoint_in_anatomical=trajectory_annotation.startpoint,
                # endpoint_in_anatomical=trajectory_annotation.endpoint,
                # principle_ray=principle_ray,
                # progress=progress,
                fractured=fractured,
                cortical_breach=cortical_breach,
                view_name="random",
                reason="no-reason",
                # view=view,
                # index_from_world=index_from_world,
                # kwire_guide_progress=-0.3, # kwire_guide_progress,
                # tau=tau,
            )

            # determine the paths for everything
            breach_str = "breach" if cortical_breach else "nobreach"
            color = np.array([212, 4, 4] if cortical_breach else [24, 201, 8]) / 255.0

            utils.save_json(self.images_dir / f"{stem}_info.json", info)
            image_utils.save(self.images_dir / f"{stem}.png", image)
            image_paths.append(self.images_dir / f"{stem}.png")
            # trajectory_annotation.save(self.trajectory_dir / f"{stem}_{reason}.mrk.json", color=color)

        return image_paths


class BayviewDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 12,
        persistent_workers: bool = False,
        dataset: Dict[str, Any] = {},
    ):
        super().__init__()
        self.loader_kwargs = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
        )
        self.dataset_kwargs = dataset
        assert "mode" not in self.dataset_kwargs

    def prepare_data(self):
        # TODO: download and process data
        # Bayview(**self.dataset_kwargs)
        self.dataset_kwargs["download"] = False

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_set = Bayview(mode="train", **self.dataset_kwargs)
            self.val_set = Bayview(mode="val", **self.dataset_kwargs)

        if stage == "test" or stage is None:
            self.test_set = Bayview(mode="test", **self.dataset_kwargs)

    def train_dataloader(
        self,
    ) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(self.train_set, shuffle=True, **self.loader_kwargs)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.val_set, **self.loader_kwargs)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.test_set, **self.loader_kwargs)
