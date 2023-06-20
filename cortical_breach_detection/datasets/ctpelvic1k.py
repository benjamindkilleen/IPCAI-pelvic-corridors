"""Datasets for the cortical breach detection task."""

from __future__ import annotations

import csv
import logging
import math
import re
from copy import deepcopy
from pathlib import Path
from shutil import rmtree
from typing import TYPE_CHECKING, Any
from typing import Dict
from typing import Generator
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Union

import subprocess as sp
from collections import Counter
from datetime import datetime
import deepdrr
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from imgaug.augmentables.lines import LineString, LineStringsOnImage
import numpy as np
import pytorch_lightning as pl
import pyvista as pv
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from deepdrr import geo
from deepdrr import vis
from deepdrr.annotations.line_annotation import LineAnnotation
from deepdrr.device import MobileCArm
from deepdrr.device.mobile_carm import pose_vector_angles
from deepdrr.utils import data_utils
from deepdrr.utils import image_utils
from imgaug.augmentables.heatmaps import HeatmapsOnImage
from PIL import Image
from rich.progress import Progress, track
from scipy.spatial import KDTree
from skimage import draw
from torch.utils.data import DataLoader

from ..viewpoint_planning import plan_viewpoint
from .. import tools
from .. import utils
from .. import augmenters
from ..loopx import LoopX
from ..console import console
from ..utils import geo_utils, eval_utils

if TYPE_CHECKING:
    from ..models import UNet

log = logging.getLogger(__name__)
DEGREE_SIGN = "\N{DEGREE SIGN}"


def get_kwire_guide(
    kwire_guide_name: Optional[str],
    kwire_guide_density: float = 0.1,
    spacer_probability: float = 0,
    **kwargs,
) -> Optional[List[tools.Tool]]:
    if kwire_guide_name is not None:
        assert hasattr(tools, kwire_guide_name), f"invalid tool name: {kwire_guide_name}"
        if kwire_guide_name in ["Disk", "Washer"]:
            kwire_guide_0 = getattr(tools, kwire_guide_name)(density=kwire_guide_density, **kwargs)
            kwire_guide_1 = getattr(tools, kwire_guide_name)(density=kwire_guide_density, **kwargs)
            kwire_guide = [kwire_guide_0, kwire_guide_1]
        elif kwire_guide_name == "WireGuide":
            kwire_guide = tools.WireGuide(
                washer_density=kwire_guide_density,
                spacer_density=kwire_guide_density * 5,
                include_spacers=np.random.rand() < spacer_probability,
                **kwargs,
            )
        else:
            kwire_guide = [getattr(tools, kwire_guide_name)(density=kwire_guide_density, **kwargs)]
    else:
        kwire_guide = None
    return kwire_guide


class CTPelvic1K:
    seg_names = [
        "kwire",
        "hip_left",
        "hip_right",
        "femur_left",
        "femur_right",
        "sacrum",
        "corridor",
    ]

    def __init__(
        self,
        root: str = "~/datasets",
        mode: Literal["train", "val", "test"] = "train",
        download: bool = False,
        generate: bool = True,
        overwrite: bool = False,
        progress_values: List[float] = [-0.1, 0.1],
        detector_distance: List[float] = [450, 700],
        num_trajectories: int = 10,
        num_views: Union[int, List[int]] = 3,
        max_view_angle: Union[float, List[float]] = [45, 5, 5],
        max_startpoint_offset: float = 8,
        max_endpoint_offset: float = 15,
        corridor_radius: float = 10,
        cortical_breach_threshold: float = 1.50,
        breach_detection_spacing: float = 0.5,
        breach_distance: float = 1.0,
        max_isocenter_offset: Union[float, List[float]] = 10,
        geodesic_threshold: float = 16,
        split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        kwire_guide: Optional[str] = None,
        kwire_guide_density: float = 0.05,
        use_cached_trajectories: bool = True,
        test_only: bool = False,
        image_size: Union[int, Tuple[int, int]] = None,
        augment: bool = True,
        disable_geometric_aug: bool = False,
        gaussian_contrast_aug: bool = True,
        device: Literal["loopx", "carm"] = "loopx",
        loopx: Dict[str, Any] = {},
        carm: Dict[str, Any] = {},
        projector: Dict[str, Any] = {},
    ):
        """Make the dataset.

        Args:
            root (str, optional): Where datasets are stored in general. Defaults to "~/datasets".
            mode (str, optional): Which dataset to use. Defaults to "train".
            download (bool, optional): Whether to download the dataset. Defaults to False.
            generate (bool, optional): Whether to generate the dataset, or just read what's already there. Defaults to True.
            overwrite (bool, optional): Whether to overwrite the dataset. Defaults to False.
            progress_values (List[float], optional): Range of progress values to sample from (uniformly) K-wire insertion. Defaults to [0.4, 0.9].
            num_trajectories (int, optional): Number of trajectories to use. Defaults to 10.
            num_views (int, optional): Number of views to use per trajectory, or a list specifying different numbers for train/val/test. Defaults to 3.
            max_view_angle (float, optional): Maximum view angle to use per view, or list for train/val/test. Defaults to [45,5,5].
            max_startpoint_offset (float, optional): Maximum startpoint offset to use for trajectories. Defaults to 8.
            max_endpoint_offset (float, optional): Maximum endpoint offset to use for trajectories. Defaults to 15.
            cortical_breach_threshold (float, optional): Density to estimate cortical breach at. Defaults to 1.50.
            breach_detection_spacing (float, optional): Spacing along the trajectory to use for breach detection. Defaults to 0.5.
            breach_distance (float, optional): Minimum distance from the centerline of the trajectory to the bone surface, for a safe traejctory. Defaults to 1.0.
            max_isocenter_offset (float, optional): Maximum isocenter offset to use when sampling views. Defaults to 10.
            geodesic_threshold (float, optional): Maximum geodesic distance to use when sampling views. Defaults to 16.
            split (Tuple[float, float, float], optional): Split of val/test/train. Defaults to (0.8, 0.1, 0.1).
            kwire_guide (str, optional): Wire Guide to use for K-wire insertion. Defaults to None.
            kwire_guide_density (float, optional): Density to use for K-wire guide voxelization. Defaults to 0.05.
            use_cached_trajectories (bool, optional): Whether to use cached trajectories. Defaults to True.
            test_only (bool, optional): Only generate the test set. Defaults to False.
            image_size (int, optional): Resize images to this size in __getitem__ for deep models. Defaults to None.
            augment (bool, optional): Whether to augment the dataset. Defaults to True.
            disable_geometric_aug (bool, optional): Whether to disable geometric augmentation. Defaults to False.
            gaussian_contrast_aug (bool, optional): Whether to use gaussian contrast augmentation. Defaults to True.
            carm (Dict[str, Any], optional): Parameters for the CARM. Defaults to {}.
            projector (Dict[str, Any], optional): Parameters for the projector. Defaults to {}.
        """

        self.root = Path(root).expanduser()
        self.dataset_dir = self.root / "CTPelvic1K"

        # Save all the parameters necessary to reproduce the dataset,
        # Just by loading this dict and calling CTPelvic1K(**config).
        config = dict(
            root=root,
            mode=mode,
            download=True,  # Stored this way to facilitate automatic download.
            overwrite=False,
            progress_values=progress_values,
            num_trajectories=num_trajectories,
            num_views=num_views,
            max_view_angle=max_view_angle,
            max_startpoint_offset=max_startpoint_offset,
            max_endpoint_offset=max_endpoint_offset,
            cortical_breach_threshold=cortical_breach_threshold,
            breach_detection_spacing=breach_detection_spacing,
            breach_distance=breach_distance,
            max_isocenter_offset=max_isocenter_offset,
            geodesic_threshold=geodesic_threshold,
            split=split,
            kwire_guide=kwire_guide,
            kwire_guide_density=kwire_guide_density,
            use_cached_trajectories=use_cached_trajectories,
            test_only=test_only,
            image_size=image_size,
            # augment=augment,
            device=device,
            loopx=loopx,
            carm=carm,
            projector=projector,
        )
        utils.save_json(self.dataset_dir / f"config.json", config)

        self.mode = mode
        self.progress_values = np.array(progress_values)
        self.detector_distance = deepdrr.utils.listify(detector_distance, 2)
        assert len(progress_values) == 2
        self.num_trajectories = num_trajectories
        self.num_views = np.array(deepdrr.utils.listify(num_views, 3))
        self.max_view_angle = np.radians(deepdrr.utils.listify(max_view_angle, 3))
        self.max_startpoint_offset = max_startpoint_offset
        self.max_endpoint_offset = max_endpoint_offset
        self.corridor_radius = corridor_radius
        self.cortical_breach_threshold = cortical_breach_threshold
        self.breach_detection_spacing = breach_detection_spacing
        self.breach_distance = breach_distance
        self.max_isocenter_offset = max_isocenter_offset
        self.geodesic_threshold = geodesic_threshold
        self.split = split
        self.kwire_guide = kwire_guide
        self.kwire_guide_density = kwire_guide_density
        self.use_cached_trajectories = use_cached_trajectories
        self.test_only = test_only
        self.image_size = deepdrr.utils.tuplify(image_size, 2)
        self.augment = augment
        self.projector_config = projector
        self.seg_projector_config = deepcopy(projector)
        self.seg_projector_config["step"] = 0.05
        self.seg_projector_config["intensity_upper_bound"] = None
        self.seg_projector_config["attenuate_outside_volume"] = False

        if device == "loopx":
            self.device = LoopX(**loopx)
        elif device == "carm":
            raise NotImplementedError("CARM not implemented yet.")
            # self.device = deepdrr.MobileCArm(**carm)
        else:
            raise RuntimeError

        # if self.test_only and self.mode != "test":
        # raise RuntimeError(f"set test_only=False for mode {mode}")

        # Raw data directories.
        self.clinic_data_dir = self.dataset_dir / "CTPelvic1K_dataset6_data"
        self.clinic_trajectories_dir = self.dataset_dir / "trajectories"
        self.segmentation_data_dir = self.dataset_dir / "ipcai2021_dataset6_Anonymized"
        self.surface_data_dir = (
            self.dataset_dir / "ipcai2021_dataset6_Anonymized_surfaces"
        )  # created by dataset
        self.totalsegmentator_dir = self.dataset_dir / "totalsegmentator"
        if not self.totalsegmentator_dir.exists():
            self.totalsegmentator_dir.mkdir()

        # Download the raw data
        if download:
            self.download()

        # Calculate things having to do with the train/val/test split.
        self.sample_paths = self._get_sample_paths()
        self.mode_indices = dict(train=0, val=1, test=2)
        self.mode_names = ["train", "val", "test"]
        self.mode_index = self.mode_indices[self.mode]

        # Calculate number of images per "thing" for each train/val/test split.
        # Dataset generated in this manner:
        # For each mode,
        # for each sample (left/right CT) in mode,
        # for each t in num_trajectories,
        # for each v in num_views[mode]:
        # sample a progress_value in progress_values, and take an image.

        self.projections_dir = self.dataset_dir / "projections"
        self.cached_trajectories_dir = self.dataset_dir / "trajectories_cache"
        if generate:
            self.samples_per_mode = utils.split_sizes(len(self.sample_paths), np.array(split))
            self.sample_modes = utils.split_mapping(len(self.sample_paths), np.array(split))
            self.images_per_trajectory = self.num_views
            self.images_per_sample = self.num_trajectories * self.images_per_trajectory
            self.images_per_mode = self.images_per_sample * self.samples_per_mode
            self.total_images = np.sum(self.images_per_mode)
            sample_dirs = self.generate(overwrite, generate=True)
        else:
            # Ignore last one, in case it is still generating.
            log.warning(
                "Not checking dataset is fully generated. Only use for debugging, when dataset is still generating."
            )
            sample_dirs = np.array(list(self.projections_dir.glob("*/"))[:-1])
            self.samples_per_mode = utils.split_sizes(len(sample_dirs), np.array(split))
            self.sample_modes = utils.split_mapping(len(sample_dirs), np.array(split))

            self.images_per_trajectory = self.num_views
            self.images_per_sample = self.num_trajectories * self.images_per_trajectory
            self.images_per_mode = self.images_per_sample * self.samples_per_mode
            self.total_images = np.sum(self.images_per_mode)

        log.info(
            f"Dataset has {self.images_per_mode[0]:,} / {self.images_per_mode[1]:,} / {self.images_per_mode[2]:,} train/val/test images"
        )
        log.info(f"projections dir: {self.projections_dir}")
        log.info(f"trajectories dir: {self.cached_trajectories_dir}")

        # Now get the images paths just for this mode.
        self.sample_modes = self.sample_modes[: len(sample_dirs)]
        sample_dirs = sample_dirs[self.sample_modes == self.mode_index]

        # log.debug(f"sample_dirs: ({len(sample_dirs)}) {sample_dirs}")

        if len(sample_dirs) == 0:
            log.warning("No images found for this mode. Did you generate them?")
            # raise RuntimeError(f"No images for mode {self.mode}")

        self.image_paths = sum(
            [self.get_images_in_sample_dir(sample_dir) for sample_dir in sample_dirs], []
        )

        # TODO: get the image paths based on the sample dirs, rather than first getting them and then splitting.
        # image_paths = self.projections_dir.glob("*/*/*_*breach.png")
        # self.image_paths = self._split_data(image_paths, mode, split)
        log.info(f"found {len(self.image_paths)} for {mode} set from {len(sample_dirs)} samples")

        self.num_negative = sum([int("nobreach" in str(p)) for p in self.image_paths])
        self.num_positive = len(self.image_paths) - self.num_negative
        self.pos_weight = self.num_negative / (self.num_positive + 0.00001)
        log.info(
            f"num positive, negative: {self.num_positive}, {self.num_negative} -> pos_weight: {self.pos_weight}"
        )

        log.info(
            f"generating image size {self.device.sensor_height, self.device.sensor_width}, resizing to {self.image_size}"
        )

        self.resize = iaa.Resize({"height": self.image_size[0], "width": self.image_size[1]})

        # NOTE: one has to be careful with geometric augmentations, since we are regressing depth.
        # Anything that changes the apparent size of the pelvis in the image is going to affect the ground truth depth.
        # TODO: get rid of Fliplr and add in Invert
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        geometric = lambda aug: iaa.Sometimes((1 - int(disable_geometric_aug)), aug)
        if_gauss = lambda aug: iaa.Sometimes((int(gaussian_contrast_aug)), aug)
        self.augmenter = iaa.Sequential(
            [
                iaa.Invert(0.5, min_value=0, max_value=1),  # invert color channels
                # apply the following augmenters to most images
                # crop images by -5% to 10% of their height/width
                geometric(
                    sometimes(
                        iaa.CropAndPad(percent=(-0.05, 0.1), pad_mode=ia.ALL, pad_cval=(0, 1))
                    )
                ),
                # geometric(
                #     sometimes(
                #         iaa.Affine(
                #             scale={"x": (0.8, 1.4), "y": (0.8, 1.4)},
                #             translate_percent={
                #                 "x": (-0.05, 0.05),
                #                 "y": (-0.05, 0.05),
                #             },  # translate by -5 to +5 percent (per axis)
                #             rotate=(-5, 5),  # rotate by -5 to +5 degrees
                #             order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                #             cval=(0, 1),  # if mode is constant, use a cval between 0 and 1
                #             mode=ia.ALL,  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                #         )
                #     )
                # ),
                # execute 0 to 5 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
                iaa.SomeOf(
                    (0, 5),
                    [
                        iaa.OneOf(
                            [
                                iaa.GaussianBlur(
                                    (0, 3.0)
                                ),  # blur images with a sigma between 0 and 3.0
                                iaa.AverageBlur(k=(2, 3)),  # blur image using local means
                                iaa.MedianBlur(k=3),  # blur image using local medians
                            ]
                        ),
                        iaa.Sharpen(alpha=(0, 0.5), lightness=(0.75, 1.5)),  # sharpen images
                        iaa.Emboss(alpha=(0, 0.5), strength=(0, 2.0)),  # emboss images
                        # search either for all edges or for directed edges,
                        # blend the result with the original image using a blobby mask
                        iaa.BlendAlphaSimplexNoise(
                            iaa.OneOf(
                                [
                                    iaa.EdgeDetect(alpha=(0, 0.5)),
                                    iaa.DirectedEdgeDetect(alpha=(0, 0.5), direction=(0.0, 1.0)),
                                ]
                            )
                        ),
                        iaa.AdditiveGaussianNoise(
                            loc=0, scale=(0.0, 0.03)
                        ),  # add gaussian noise to images
                        iaa.OneOf(
                            [
                                iaa.Dropout((0.01, 0.1)),  # randomly remove up to 10% of the pixels
                                iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05)),
                            ]
                        ),
                        # either change the brightness of the whole image (sometimes
                        # per channel) or change the brightness of subareas
                        iaa.Add((-0.03, 0.01)),  # change brightness of images
                        # iaa.OneOf(
                        #     [
                        #         iaa.LinearContrast((0.8, 1.2)),  # improve or worsen the contrast
                        #         iaa.AllChannelsCLAHE(clip_limit=(1, 10)),
                        #     ]
                        # ),
                        # Non-uniform contrast adjustment
                        if_gauss(augmenters.gaussian_contrast(alpha=(0.1, 0.8), sigma=(0.1, 0.5))),
                        # move pixels locally around
                        # geometric(
                        #     sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25))
                        # ),
                        # move parts of the image around
                        # geometric(sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05)))),
                        # geometric(sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))),
                    ],
                    random_order=True,
                ),
            ],
            random_order=True,
        )

    def _get_sample_paths(self) -> List[Tuple[Path, Path]]:
        """Get the list of (CT, annotation) pairs in the dataset."""
        sample_paths = []
        for ct_path in sorted(list(self.clinic_data_dir.glob("*.nii.gz"))):
            base = self.get_base_from_ct_path(ct_path)
            if base is None:
                continue
            annotation_paths = self.clinic_trajectories_dir.glob(f"{base}*.mrk.json")
            annotation_paths = sorted(list(annotation_paths))
            for annotation_path in annotation_paths:
                sample_paths.append((ct_path, annotation_path))

        # TODO: same thing for NMDID data.

        return sample_paths

    def _split_data(
        self,
        image_paths: List[Path],
        mode: Literal["train", "val", "test"],
        split: Tuple[float, float, float],
    ):
        """DEPRECATED: assumes equal images per ct in each dataset."""
        if self.test_only:
            if mode == "test":
                return image_paths
            else:
                return []

        image_paths = list(sorted(image_paths))
        num_samples = len(self.sample_paths)
        assert num_samples * self.images_per_sample == len(
            image_paths
        ), f"{num_samples} * {self.images_per_sample} != {len(image_paths)}. Maybe regenerate the dataset with new trajectories?"

        indices = utils.split_indices(num_samples, np.array(split))
        indices *= self.images_per_sample

        idx = dict(train=0, val=1, test=2)[mode]
        return image_paths[indices[idx] : indices[idx + 1]]

    def download(self):
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        if not self.clinic_data_dir.exists():
            data_utils.download(
                url="https://zenodo.org/record/4588403/files/CTPelvic1K_dataset6_data.tar.gz?download=1",
                filename="CTPelvic1K_dataset6_data.tar.gz",
                root=self.dataset_dir,
                md5="6b6121e3094cb97bc452db99dd1abf56",
                extract_name="CTPelvic1K_dataset6_data",
            )
            data_utils.download(
                url="https://zenodo.org/record/4588403/files/CTPelvic1K_dataset6_Anonymized_mask.tar.gz?download=1",
                filename="CTPelvic1K_dataset6_Anonymized_mask.tar.gz",
                root=self.dataset_dir,
                md5="7696eb7c4da91f989d6842d7b422efd9",
                extract_name="CTPelvic1K_dataset6_data",
            )
        if not self.clinic_trajectories_dir.exists():
            data_utils.download(
                url="https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/bkillee1_jh_edu/EWN6n7wn8mFCobtdSdXPPsYBZuyyvfcbnqYsw2KtYLRvNw?e=2b1gJL&download=1",
                filename="trajectories.zip",
                root=self.dataset_dir,
                md5="d0f66a9a271b1fb5c576a6cca239c8dd",
                extract_name="trajectories",
            )

    def _get_image_stem(self, image_path: Path):
        """Get the stem of the image path."""
        return Path(image_path).stem[:-4]  # Cut off the _drr suffix.

    def _get_image_info(self, image_path: Path):
        stem = self._get_image_stem(image_path)
        return utils.load_json(image_path.parent / f"{stem}_info.json")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]

        # get the insertion depth for metrics
        image_info = self._get_image_info(image_path)
        startpoint = geo.point(image_info["startpoint"])
        endpoint = geo.point(image_info["endpoint"])
        progress = float(image_info["progress"])
        insertion_depth = progress * (endpoint - startpoint).norm()

        # Move the C-arm just in case we need the transforms
        self.device.set_config(image_info["device"])
        index_from_world = self.device.index_from_world
        camera3d_from_world = self.device.camera3d_from_world
        source_to_detector_distance = self.device.source_to_detector_distance
        # index_from_world = geo.CameraProjection(*image_info["index_from_world"])
        # camera3d_from_world = geo.frame_transform(image_info["camera3d_from_world"])
        # source_to_detector_distance = self.device.source_to_detector_distance
        # source_to_detector_distance = image_info["source_to_detector_distance"]

        # get the image and cortical breach label
        image = np.array(Image.open(image_path)).astype(np.float32) / 255.0
        image = np.atleast_3d(image)
        H, W = image.shape[:2]
        _H, _W = H, W

        # All in the world frame, unless specified
        gt_startpoint: geo.Point3D = geo.point(image_info["gt_startpoint"])
        gt_endpoint: geo.Point3D = geo.point(image_info["gt_endpoint"])
        gt_startpoint_in_index = index_from_world @ gt_startpoint
        gt_endpoint_in_index = index_from_world @ gt_endpoint
        gt_startpoint_in_camera = camera3d_from_world @ gt_startpoint
        gt_endpoint_in_camera = camera3d_from_world @ gt_endpoint
        a_x = int(gt_startpoint_in_index[0])
        a_y = int(gt_startpoint_in_index[1])
        b_x = int(gt_endpoint_in_index[0])
        b_y = int(gt_endpoint_in_index[1])
        startpoint_is_in_image = (0 <= a_y < H) and (0 <= a_x < W)

        # Source to startpoint distance scaled to [0, 1]
        source_to_startpoint_scaled = gt_startpoint_in_camera.z / source_to_detector_distance
        source_to_startpoint_mm = gt_startpoint_in_camera.z

        # TODO: consider having both left/right in image
        keypoints = [
            Keypoint(x=a_x, y=a_y),
        ]
        lines = [
            LineString([(a_x, a_y), (b_x, b_y)]),
        ]

        # Get the heatmap of the startpoint.
        # startpoint_heatmap = utils.nn_utils.heatmap_numpy(a_i, a_j, scale=H / 80, size=(H, W))
        # corridor_heatmap = utils.nn_utils.line_segment_heatmap_numpy(
        # a_i, a_j, b_i, b_j, scale=H / 80, size=(H, W)
        # )
        # heatmaps = np.stack([startpoint_heatmap, corridor_heatmap], axis=2)

        # phi, for metrics purposes
        v = gt_endpoint_in_index - gt_startpoint_in_index
        phi = np.arctan2(v[0], v[1])

        stem = self._get_image_stem(image_path)
        if self.kwire_guide is not None:
            # guide segmentation
            guide_mask_path = image_path.parent / f"{stem}_kwire-guide-mask.png"
            kwire_guide_mask = np.array(Image.open(guide_mask_path))
            kwire_guide_mask = (kwire_guide_mask > 0).astype(np.int32)
            raise NotImplementedError("kwire guide not implemented")
        else:
            kwire_guide_mask = np.zeros((H, W), dtype=np.int32)

        segs = []
        for seg_name in self.seg_names:
            seg_path = list(image_path.parent.glob(f"{stem}_{seg_name}*.png"))[0]
            seg = np.array(Image.open(seg_path))
            seg = (seg > 0).astype(np.int32)
            segs.append(seg)

        # combine segmentations, with the kwire taking priority
        segs = np.stack(segs, axis=2)

        if self.image_size[0] != H and self.image_size[1] != W:
            # Resize to desired resolution for model
            images_resized, segs_resized, keypoints_resized, lines_resized = self.resize(
                images=image[None],
                segmentation_maps=segs[None],
                keypoints=[keypoints],
                line_strings=[lines],
            )
            image = images_resized[0]
            segs = segs_resized[0]
            keypoints = keypoints_resized[0]
            lines = lines_resized[0]
            H = self.image_size[0]
            W = self.image_size[1]

            # raise NotImplementedError("make sure mask and other things resized")

        # augmentation
        if self.mode in ["train"] and self.augment:
            # Make images/heatmaps/segmentations for imgaug
            imgaug_images = image[None, :, :, :]
            imgaug_segs = segs[None, :, :, :]

            # Augment
            images_aug, segs_aug, keypoints_aug, lines_aug = self.augmenter(
                images=imgaug_images,
                segmentation_maps=imgaug_segs,
                keypoints=[keypoints],
                line_strings=[lines],
            )

            # Get images/heatmaps/segmentations back out
            image = images_aug[0]
            segs = segs_aug[0, :, :, :]
            keypoints = keypoints_aug[0]
            lines = lines_aug[0]

        # Convert keypoints heatmaps
        left_side = "left" in image_info["annotation"]
        kpsoi = KeypointsOnImage(keypoints, shape=image.shape)
        scale = 2 / self.device.pixel_size  # 2mm
        startpoint_heatmap = np.exp(-np.square(kpsoi.to_distance_maps()) / (2 * scale**2))

        # Depth and valid maps
        a_x = int(keypoints[0].x)
        a_y = int(keypoints[0].y)
        depth_maps = np.zeros((1, H, W), dtype=np.float32)
        valid_maps = np.zeros((1, H, W), dtype=np.float32)
        if (0 <= a_y < H) and (0 <= a_x < W):
            depth_maps[0, a_y, a_x] = source_to_startpoint_scaled
            valid_maps[0, a_y, a_x] = 1

        # Convert to tensors
        image = torch.tensor(image).permute(2, 0, 1)
        segs = torch.tensor(segs).permute(2, 0, 1)
        # log.debug(f"segs.shape: {segs.shape}")
        startpoint_heatmaps = torch.tensor(startpoint_heatmap).permute(2, 0, 1)

        cortical_breach = image_info["cortical_breach"]
        cortical_breach_label = torch.tensor([cortical_breach], dtype=torch.int64)
        target = dict(
            segs=segs[0:6, :, :],
            corridor_masks=segs[6:7, :, :],
            startpoint_heatmaps=startpoint_heatmaps,
            source_to_startpoint_scaled=torch.tensor(
                source_to_startpoint_scaled, dtype=torch.float32
            ),
            depth_maps=depth_maps,
            valid_maps=valid_maps,
        )

        # Get the base name
        base = image_info.get(
            "base",
            self.get_base_from_ct_path(image_info["ct"]) + ("_left" if left_side else "_right"),
        )

        info = dict(
            cortical_breach_label=cortical_breach_label,
            left_side=torch.tensor(left_side, dtype=torch.long),
            image_path=str(image_path),
            insertion_depth=insertion_depth,
            progress=torch.tensor(image_info["progress"], dtype=torch.float32),
            startpoint=torch.tensor(np.array(startpoint), dtype=torch.float32),
            endpoint=torch.tensor(np.array(endpoint), dtype=torch.float32),
            ct_path=image_info["ct"],
            annotation_path=image_info["annotation_path"],
            base=base,
            camera3d_from_world=torch.tensor(geo.get_data(camera3d_from_world)),
            index_from_world=torch.tensor(geo.get_data(index_from_world)),
            world_from_index=torch.tensor(geo.get_data(index_from_world.inv)),
            gt_startpoint=torch.tensor(np.array(gt_startpoint), dtype=torch.float32),
            gt_endpoint=torch.tensor(np.array(gt_endpoint), dtype=torch.float32),
            phi=torch.tensor(phi, dtype=torch.float32),
            startpoint_is_in_image=torch.tensor(startpoint_is_in_image, dtype=bool),
            source_to_detector_distance=torch.tensor(
                self.device.source_to_detector_distance, dtype=torch.float32
            ),
            pixel_size=torch.tensor(self.device.pixel_size, dtype=torch.float32),
            a_x=torch.tensor(a_x, dtype=torch.long),
            a_y=torch.tensor(a_y, dtype=torch.long),
            image_height=torch.tensor(_H, dtype=torch.long),
            image_width=torch.tensor(_W, dtype=torch.long),
            source_to_startpoint_mm=torch.tensor(source_to_startpoint_mm, dtype=torch.float32),
        )

        # if self.mode == "test":
        #     world_from_device = geo.frame_transform(image_info_devi)
        #     # info["guide_density"] = torch.tensor(image_info["guide_density"], dtype=torch.float32)
        #     info["kwire_density"] = torch.tensor(image_info["kwire_density"], dtype=torch.float32)
        #     info["kwire_radius"] = torch.tensor(image_info["kwire_radius"], dtype=torch.float32)
        #     info["q"] = torch.tensor(image_info["device"]["q"], dtype=torch.float32)
        #     info["world_from_device"] = torch.tensor(image_info["device"]["world_from_device"])
        #     info["anatomical_from_world"] = torch.tensor(image_info["anatomical_from_world"])

        return image, target, info

    @staticmethod
    def get_base_from_ct_path(ct_path: Union[Path, str]) -> Optional[str]:
        """Get a base stem to use for annotation or image paths, corresponding to the CT.

        Args:
            ct_path (Path): Posixpath to the ct.

        Returns:
            Optional[str]: Either the base string, or None if parsing fails.
        """

        pattern = r"(?P<base>dataset6_CLINIC_\d+)_data\.nii\.gz"
        if (m := re.match(pattern, Path(ct_path).name)) is None:
            return None
        else:
            return m.group("base")

    def get_views(
        self,
        annotation: deepdrr.LineAnnotation,
        ct: deepdrr.Volume,
        mode: Literal["train", "val", "test"],
    ) -> Generator[Tuple[geo.Point3D, geo.Vector3D], None, None]:
        """Get a distribution of views onto the anatomy.

        Args:
            annotation (deepdrr.LineAnnotation): The groundtruth annotation.
            ct (deepdrr.Volume): The CT volume.
            mode (str): train/val/test mode this sample belongs to.

        Yields:
            Tuple[geo.Point3D, geo.Vector3D]: A tuple of the point of interest and view direction
                (pointing toward detector).
        """

        assert annotation.anatomical_coordinate_system == "RAS"

        startpoint = annotation.startpoint_in_world
        endpoint = annotation.endpoint_in_world
        mode_idx = self.mode_indices[mode]
        n = self.num_views[mode_idx]
        rays = geo.random.spherical_uniform(
            center=geo.v(0, 0, 1), d_phi=self.max_view_angle[mode_idx], n=n
        )
        scale = np.array([15, 30, 30])
        for ray in rays:
            point = startpoint.lerp(endpoint, np.random.uniform(-0.05, 0.9))
            offset = np.clip(np.random.normal(scale=scale, size=3), -2 * scale, 2 * scale)
            point = point + geo.v(*offset)
            yield point, ray

    def get_first_views(
        self, annotation: deepdrr.LineAnnotation, images_per_view: int = 5
    ) -> Generator[Tuple[str, geo.Point3D, geo.Vector3D], None, None]:
        """Get a distribution of views for testing triangulation.

        We will test n of each views: AP, inlet, and outlet. The center will be sampled same as in training data.

        Args:
            annotation (deepdrr.LineAnnotation): The groundtruth annotation for the corridor.
            images_per_view (int, optional): Number of images to generate per view. Defaults to 5.

        """
        assert annotation.anatomical_coordinate_system == "RAS"
        startpoint = annotation.startpoint_in_world
        endpoint = annotation.endpoint_in_world

        # In device coordinates, assuming patient is head first, supine
        views = {
            "AP": geo.v(0, 0, 1),
            "Inlet": geo.v(0, 0, 1).rotate(geo.v(1, 0, 0), np.radians(15)),
            "Outlet": geo.v(0, 0, 1).rotate(geo.v(1, 0, 0), np.radians(-15)),
        }

        scale = np.array([5, 5, 5])
        for name in ["AP"]:
            rays = geo.random.spherical_uniform(
                center=views[name], d_phi=np.radians(5), n=images_per_view
            )
            for ray in rays:
                point = startpoint.lerp(endpoint, np.random.uniform(0.2, 0.4))
                # offset = np.clip(np.random.normal(scale=scale, size=3), -2 * scale, 2 * scale)
                # point = point + geo.v(*offset)
                yield name, point, ray

    def _check_projections_dir(self, overwrite: bool = False) -> bool:
        if self.projections_dir.exists() and not overwrite:
            log.info(
                f"Continuing with dataset at {self.projections_dir}. Set dataset.overwrite=true to completely overwrite it."
            )
            return False

        if self.projections_dir.exists():
            log.critical(
                f"DELETING {self.projections_dir}!\nPress ENTER to continue, Ctrl-C to cancel..."
            )
            input()
            # sleep(5)
            rmtree(self.projections_dir)

        self.projections_dir.mkdir()
        return False

    def get_ct(
        self,
        ct_path: str,
        left_side: bool,
        world_from_anatomical: Optional[geo.F] = None,
        imaging_center_in_world: Optional[geo.Point3D] = None,
        use_material_segmentations: Literal["always", "never", "random"] = "always",
    ) -> Tuple[
        deepdrr.Volume, Dict[str, deepdrr.Volume], Dict[str, pv.PolyData], Dict[str, KDTree]
    ]:
        """Get the CT, segmentations, surfaces, and KDTree for the surfaces.

        Args:
            ct_path (str): Path to the CT.
            left_side (bool): Whether annotation is for the left side.
            imaging_center_in_world (Optional[geo.Point3D], optional): Imaging center in world coordinates. Defaults to None.
            use_material_segmentations (Literal["always", "never", "random"], optional): Whether to use the organ segmentations
                to fix segmentations of bone in DeepDRR. Defaults to "always".

        """
        ct = deepdrr.Volume.from_nifti(ct_path, use_thresholding=True, use_cached=False)
        if world_from_anatomical is not None:
            ct.world_from_anatomical = world_from_anatomical
        elif imaging_center_in_world is not None:
            ct.orient_patient(head_first=True, supine=True)
            ct.place_center(imaging_center_in_world)

        # Remove the calibration bar from the CT segmentation.
        for m in ct.materials.keys():
            ct.materials[m][:, 400:, :] = 0

        base = CTPelvic1K.get_base_from_ct_path(ct_path)
        dataset_dir = Path(ct_path).parent.parent

        # Run TotalSegmentator:
        seg_dir = self.totalsegmentator_dir / base
        if seg_dir.exists() and len(list(seg_dir.glob("*.nii.gz"))) == 104:
            pass
        else:
            seg_dir.mkdir(exist_ok=True)
            log.info(f"Running Total Segmentator for {base}")
            # run totalsegmentator
            sp.run(
                [
                    "TotalSegmentator",
                    "-i",
                    str(ct_path),
                    "-o",
                    str(seg_dir),
                ]
            )

        # load the segmentations
        # seg_path = dataset_dir / "ipcai2021_dataset6_Anonymized" / f"{base}_mask_4label.nii.gz"
        # seg = deepdrr.Volume.from_nifti(
        #     seg_path, segmentation=True, world_from_anatomical=ct.world_from_anatomical,
        # )
        segs: Dict[str, deepdrr.Volume] = {}
        segs["hip_left"] = deepdrr.Volume.from_nifti(
            seg_dir / "hip_left.nii.gz",
            segmentation=True,
            world_from_anatomical=ct.world_from_anatomical,
        )
        segs["hip_right"] = deepdrr.Volume.from_nifti(
            seg_dir / "hip_right.nii.gz",
            segmentation=True,
            world_from_anatomical=ct.world_from_anatomical,
        )
        segs["femur_left"] = deepdrr.Volume.from_nifti(
            seg_dir / "femur_left.nii.gz",
            segmentation=True,
            world_from_anatomical=ct.world_from_anatomical,
        )
        segs["femur_right"] = deepdrr.Volume.from_nifti(
            seg_dir / "femur_right.nii.gz",
            segmentation=True,
            world_from_anatomical=ct.world_from_anatomical,
        )
        segs["sacrum"] = deepdrr.Volume.from_nifti(
            seg_dir / "sacrum.nii.gz",
            segmentation=True,
            world_from_anatomical=ct.world_from_anatomical,
        )

        # Randomly use the available segmentation to refine the DeeDRR material segmentation of the bone.
        if use_material_segmentations == "always":
            use_material_segmentations = True
        elif use_material_segmentations == "never":
            use_material_segmentations = False
        elif use_material_segmentations == "random":
            use_material_segmentations = np.random.choice([True, False])
        else:
            raise ValueError(f"Invalid use_material_segmentations={use_material_segmentations}")

        for seg in segs.values():
            mask = seg.data > 0
            if use_material_segmentations:
                ct.materials["bone"] = np.logical_or(ct.materials["bone"], mask).astype(np.float32)

        surfaces_dir = dataset_dir / "surfaces"
        if not surfaces_dir.exists():
            surfaces_dir.mkdir()

        left_side_str = "left" if left_side else "right"
        surfaces = {}
        trees = {}
        trees_in_anatomical = {}
        for name, seg in segs.items():
            surface_dir = dataset_dir / "surfaces" / base
            if not surface_dir.exists():
                surface_dir.mkdir()
            surface_path = surface_dir / f"{base}_{left_side_str}_{name}.stl"

            # Get the surface in world coordinates. Note that it is saved in LPS coordinates.
            log.debug(f"Loading surface from {surface_path}")
            if surface_path.exists():
                surface = pv.read(surface_path)
                surface.transform(geo.RAS_from_LPS.get_data(), inplace=True)
            else:
                log.debug(f"Running marching cubes on {ct_path}")
                surface = seg.isosurface(0.5)
                surf: pv.PolyData = surface.copy()
                surf.transform(geo.LPS_from_RAS.get_data(), inplace=True)
                surf.save(surface_path)
            surface.transform(ct.world_from_anatomical.get_data(), inplace=True)
            surfaces[name] = surface
            trees[name] = KDTree(surface.points.astype(np.double))

        return ct, segs, surfaces, trees

    def _get_corridor(
        self,
        annotation: LineAnnotation,
        hip_seg: deepdrr.Volume,
        hip_tree: KDTree,
        femur_tree: KDTree,
    ) -> deepdrr.Volume:
        """Get a volume for the region around the annotation.

        This is restricted to be 2mm from the surface of the hip and 5mm from the femoral head.

        """
        log.info(f"Getting corridor...")
        points_in_IJK = geo.core._to_homogeneous(np.argwhere(hip_seg.data > 0.5))
        log.debug(f"points_in_IJK.shape = {points_in_IJK.shape}")
        points = (np.array(hip_seg.world_from_ijk) @ points_in_IJK.T).T
        points = points[:, :3]

        # annotation_mesh = annotation.get_mesh()
        # hip_mesh = pv.PolyData(hip_tree.data)
        # plotter = pv.Plotter()
        # plotter.add_mesh(annotation_mesh, color="red")
        # plotter.add_mesh(hip_mesh, color="white", opacity=0.5)
        # plotter.show()

        log.info(f"limiting to points near corridor for {len(points)} points")
        l = geo.line(annotation.startpoint_in_world, annotation.endpoint_in_world)
        p = np.array(l.get_point())
        v = np.array(l.get_direction())
        diff = points - p
        log.debug(f"diff.shape = {diff.shape}")
        distance_to_corridor = np.linalg.norm(diff - (diff @ v)[:, None] * v, axis=1)
        log.debug(
            f"distance_to_corridor: {distance_to_corridor.min()} - {distance_to_corridor.max()}"
        )
        points = points[distance_to_corridor < self.corridor_radius]
        log.info(f"query distance to hip surface for {len(points)} points")
        distance_to_hip_surface = hip_tree.query(points)[0]
        points = points[distance_to_hip_surface > 2]

        log.info(f"query distance to femur for {len(points)} points")
        distance_to_femur = femur_tree.query(points)[0]
        points = points[distance_to_femur > 5]

        points_in_IJK = (
            np.array(hip_seg.ijk_from_world) @ geo.core._to_homogeneous(points).T
        ).T.astype(np.int64)[:, :3]
        data = np.zeros_like(hip_seg.data)
        data[points_in_IJK[:, 0], points_in_IJK[:, 1], points_in_IJK[:, 2]] = 1
        return deepdrr.Volume(
            data,
            materials=dict(bone=data.astype(bool)),
            anatomical_from_IJK=hip_seg.anatomical_from_ijk,
            world_from_anatomical=hip_seg.world_from_anatomical,
            anatomical_coordinate_system=hip_seg.anatomical_coordinate_system,
        )

    def get_corridor(
        self,
        annotation: LineAnnotation,
        hip_seg: deepdrr.Volume,
        hip_tree: KDTree,
        femur_tree: KDTree,
        corridor_path: Path,
    ) -> deepdrr.Volume:

        if corridor_path.exists() and False:
            # This isn't actually that expensive, so don't bother using the cache.
            corridor = deepdrr.Volume.load(corridor_path)
        else:
            corridor = self._get_corridor(
                annotation,
                hip_seg,
                hip_tree,
                femur_tree,
            )
            corridor.save(corridor_path)
        return corridor

    def load_annotations(self, annotation_path, ct):
        return deepdrr.LineAnnotation.from_markup(annotation_path, ct)

    def sample_trajectory(
        self, annotation: LineAnnotation, tree: KDTree
    ) -> Tuple[geo.Point3D, geo.Point3D]:
        """Sample trajectory.

        Args:
            annotation (LineAnnotation): [description]
            tree (KDTree): KDTree of the points on the surface, in world coordinates.

        Returns:
            Tuple[geo.Point3D, geo.Point3D]: the startpoint and endpoint of the new trajectory in world coordinates.
        """
        trajectory = annotation.endpoint_in_world - annotation.startpoint_in_world
        startpoint_offset_direction = trajectory.perpendicular(random=True)
        endpoint_offset_direction = trajectory.perpendicular(random=True)

        startpoint_offset = np.clip(
            np.random.normal(0, self.max_startpoint_offset / 2),
            -self.max_startpoint_offset,
            self.max_startpoint_offset,
        )
        endpoint_offset = np.clip(
            np.random.normal(0, self.max_endpoint_offset / 2),
            -self.max_endpoint_offset,
            self.max_endpoint_offset,
        )
        startpoint = (
            annotation.startpoint_in_world + startpoint_offset * startpoint_offset_direction
        )
        endpoint = annotation.endpoint_in_world + endpoint_offset * endpoint_offset_direction

        # Only sampling trajectories with offset orthogonal to principle ray
        # Move the startpoint to the closest point on the surface
        d, idx = tree.query(np.array(startpoint))
        if d > 10:
            log.critical(f"closest point on surface is {d}mm away")
        startpoint = geo.point(tree.data[idx].copy())

        return startpoint, endpoint

    @staticmethod
    def detect_cortical_breach(
        ct: deepdrr.Volume,
        seg: deepdrr.Volume,
        surface: pv.PolyData,
        tree: KDTree,
        annotation: LineAnnotation,
        startpoint: geo.Point3D,
        endpoint: geo.Point3D,
        fractured: bool = False,
        breach_detection_spacing: float = 0.5,
        breach_distance: float = 1,
        geodesic_threshold: float = 16,
        cortical_breach_threshold: float = 1.5,
        initial_point_uncertain: bool = False,
    ) -> Tuple[bool, str]:
        """Decide whether the values correspond to a cortical breach.

        1. Use points.select_enclosed_points for points along the trajectory to get the regions
           where the center line is inside the bone. (Equivalent to interpolating the segmentation.)
        2. if the whole centerline is in bone, evaluate the closest point to the surface along its
           length. If this falls below 1mm, return True. Otherwise, return False.
        3. If there is only one exit point, then get the geodesic distance between that point and
           the endpoint of the GT annotation. If this is less than 16 mm, then it is
           likely on the same back surface of the pelvis, and is fine (return False). Otherwise, it
           may be in the acetabulum, and is a breach (return True).
        4. If there are two exit points, run (4) on the last one. The first one may be a fracture or
           it may be a breach. Obtain the average cortical bone density by querying the density of
           the CT at the first few points along the entry, as well as a couple other points closest
           to the inital 10-20 mm of the trajectory (but still on the surface). Then look at the ct
           density over the breach. If it is sharply peaked and exceeds this value, then return
           True. Otherwise return False.
        5. If there are more than two exit points, return True.

        Args:
            ct:
            seg:
            surface:
            tree:
            annotation (LineAnnotation): The original line annotation, which should be (close) to correct.
            startpoint (geo.Point3D): Startpoint of the trajectory, in world coordinates.
            endpoint (geo.Point3D): Endpoint of the trajectory, in world coordinates.
            fractured: not used.

        Returns:
            Tuple[bool, str]: Whether the trajectory is unsuitable (True) or suitable (False), i.e. whether it breaches cortex, and the detected reason.
        """

        # Sanity check on the angle.
        trajectory = endpoint - startpoint
        if trajectory.angle(
            annotation.endpoint_in_world - annotation.startpoint_in_world
        ) > np.radians(15):
            return True, "large_angle"

        num_points = int((endpoint - startpoint).norm() / breach_detection_spacing)
        points = [startpoint.lerp(endpoint, p) for p in np.linspace(0, 1, num_points)]
        # TODO: make sure surface is closed.
        # if surface.n_open_edges > 0:
        #     surface = pv.wrap(surface.points).reconstruct_surface()
        # inside_surface = pv_points.select_enclosed_points(surface, check_surface=True)["SelectedPoints"]
        inside_surface = seg.interpolate(*points) > 0.5
        log.debug(f"inside_surface: {inside_surface}")

        if initial_point_uncertain:
            idx = np.min(np.where(inside_surface)[0])
            points = points[idx:]
            inside_surface = inside_surface[idx:]
            log.debug(f"inside_surface: {inside_surface}")
            num_points = len(points)

        # Set the first index to "inside" so it doesn't interfere with later assumptions.
        inside_surface[0] = 1

        log.debug(f"inside_surface:\n{inside_surface.astype(int)}")

        closest_vert_dist, closest_vert_indices = tree.query(points)
        close_to_surface = closest_vert_dist < breach_distance
        log.debug(f"close_to_surface:\n{close_to_surface.astype(int)}")

        # Get the indices where a 1 is followed by a 0. Since the first point must be a one, any second exit point must have a corresponding entry point.
        exit_point_indices = utils.find_pattern(inside_surface, [1, 0])
        entry_point_indices = utils.find_pattern(inside_surface, [0, 1])
        log.debug(f"exit_point_indices: {exit_point_indices}")
        log.debug(f"entry_point_indices: {entry_point_indices}")

        # Check that while the trajectory is inside the bone (and far enough away from the
        # entry/exit points) it is also far enough away from the surface.
        breach = False
        reason = "good"
        breach_distance_idx = int((breach_distance + 3) / breach_detection_spacing) + 1
        for i, exit_idx in enumerate(exit_point_indices):
            if i == 0:
                # Checking the trajectory up to the first exit.
                entry_idx = 0
                local_close_to_surface = close_to_surface[
                    2 * breach_distance_idx : exit_idx - breach_distance_idx
                ]
                breach = np.any(local_close_to_surface)
                reason = "prox-1"
            elif i > 0 and len(entry_point_indices) > i - 1:
                # Checking the trajectory in between an entry and an exit
                entry_idx = entry_point_indices[i - 1]  # the one before this exit point
                local_close_to_surface = close_to_surface[
                    entry_idx + breach_distance_idx : exit_idx - breach_distance_idx
                ]
                breach = np.any(local_close_to_surface)
                reason = "prox-2"
            else:
                log.warning(
                    f"The trajectory has exited, but there was no entry point before it. Error?"
                )
                raise RuntimeError

            if breach:
                log.debug(
                    f"detected breach due to proximity to surface in {entry_idx} - {exit_idx}, with local_close_to_surface:\n{local_close_to_surface}"
                )
                return True, reason

        if exit_point_indices and not bool(inside_surface[-1]):
            # Regardless of any internal exits, check the final exit point with the geodesic.
            exit_point = np.array(points[exit_point_indices[-1]])
            gt_exit_point = np.array(annotation.endpoint_in_world)
            log.debug(f"comparing {exit_point} with gt {gt_exit_point} for geodesic")
            _, indices = tree.query(np.stack([exit_point, gt_exit_point], axis=0))
            if indices[0] != indices[1]:
                # I think the indices may be equal
                log.debug(
                    f"Taking geodesic distance for indices {indices} corresponding to {surface.points[indices]}"
                )
                try:
                    dist = surface.geodesic_distance(indices[0], indices[1])
                except ValueError:
                    return True, "geodesic-fail"

                if dist > geodesic_threshold:
                    log.debug(
                        f"Detected breach due to exit point in wrong region with dist: {dist}"
                    )
                    return True, "geodesic-dist"

        if len(exit_point_indices) == 0:
            # If there are no internal exits, only need to check the distance to surface beyond the initial entry
            # (and we don't care about the last millimeter either).
            return (
                np.any(close_to_surface[2 * breach_distance_idx : -breach_distance_idx]),
                "prox-3",
            )
        elif len(exit_point_indices) == 1 and len(entry_point_indices) == 0:
            # Should still be good. Exit was in a fine region, due to above check.
            return False, "good"
        elif len(exit_point_indices) in [1, 2] and len(entry_point_indices) == 1:
            if not fractured:
                return True, "exits-1"

            exit_idx = exit_point_indices[0]
            entry_idx = entry_point_indices[0]
            assert entry_idx > exit_idx
            if breach_detection_spacing * (entry_idx - exit_idx) > 20:
                return True, "longfrac"

            # Check the exit to see if it's in a fracture.
            assert len(entry_point_indices) == 1, "logic error with patterns"
            ct_values = ct.interpolate(*points)
            log.debug(
                f"ct min, max: {ct.data.max(), ct.data.min()}; 80, 90 percentile: {np.percentile(ct.data, [80, 90])}"
            )
            log.debug(
                f"ct_values along trajectory: {ct_values.min(), ct_values.max()}\n{ct_values}"
            )
            log.debug(f"at exit_point: {ct_values[exit_point_indices[0]]}")
            log.debug(f"at entry_point: {ct_values[entry_point_indices[0]]}")
            # TODO: also check region around each exit/entry point

            width = 3
            return (
                np.any(
                    ct_values[exit_idx - width : exit_idx + width + 1] >= cortical_breach_threshold
                )
                or np.any(
                    ct_values[entry_idx - width : entry_idx + width + 1]
                    >= cortical_breach_threshold
                ),
                f"ct-value",
            )
        else:
            return True, "exits-2"

    @staticmethod
    def get_images_in_sample_dir(sample_dir: Path) -> List[Path]:
        """Get the images (not the segmentations) in the sample dir.

        Returns:
            List[Path]: list of image paths, not necessarily sorted.
        """
        return list(Path(sample_dir).glob("*/*_drr.png"))

    def get_trajectory(
        self,
        trajectory_path: Path,
        annotation: LineAnnotation,
        tree: KDTree,
        ct: deepdrr.Volume,
        seg: deepdrr.Volume,
        surface: pv.PolyData,
        fractured: bool,
    ) -> Tuple[LineAnnotation, bool, str, bool]:
        """Get the cached, perturbed trajectory, or sample a new one and determine cortical breach.

        Args:
            trajectory_path (Path): The path to look for the trajectory.
            annotation (LineAnnotation): The original ground truth annotation to perturb from.
            tree (KDTree): A KDTree of points on the surface of the bone.
            ct (deepdrr.Volume): The CT of the pelvis.
            seg (deepdrr.Volume): The segmentation of the pelvis.
            surface (pv.PolyData): The surface of the pelvis corresponding to the tree.
            fractured (bool): Whether or not this side of the pelvis is fractured on the superior pubic ramus.

        Returns:
            Tuple[LineAnnotation, bool, str, bool]: The perturbed_annotation, cortical breach GT, reason for breach decision, and whether to save the trajectory.
        """
        # Get the trajectory
        if self.use_cached_trajectories and trajectory_path is not None:
            cortical_breach = trajectory_path.name.split("_")[1] == "breach"
            reason = trajectory_path.name.split("_")[2]
            trajectory_annotation = LineAnnotation.from_markup(trajectory_path, volume=ct)
            save_trajectory = False
        else:
            # Get trajectory and evaluation for cortical breach.
            startpoint, endpoint = self.sample_trajectory(annotation, tree)  # in world coordinates
            trajectory_annotation = LineAnnotation(
                ct.anatomical_from_world @ startpoint, ct.anatomical_from_world @ endpoint, ct
            )

            # Save the trajectory before checking it, for debugging.
            # trajectory_annotation.save(trajectory_dir / f"tmp.mrk.json", color=np.array([215, 9, 230]) / 255)

            # Determine if trajectory will pierce the cortex
            cortical_breach, reason = self.detect_cortical_breach(
                ct,
                seg,
                surface,
                tree,
                annotation,
                startpoint,
                endpoint,
                fractured=fractured,
                breach_detection_spacing=self.breach_detection_spacing,
                breach_distance=self.breach_distance,
                geodesic_threshold=self.geodesic_threshold,
                cortical_breach_threshold=self.cortical_breach_threshold,
            )
            save_trajectory = True

        return trajectory_annotation, cortical_breach, reason, save_trajectory

    def generate(self, overwrite: bool = False, generate: bool = True) -> np.ndarray:
        """Generate the dataset using deepdrr.

        Args:
            overwrite (bool, optional): Whether to overwrite existing data. Defaults to False.

        Returns:
            np.ndarray: Array containing the paths to the image directories for each sample. Only returns those with images in them.
        """
        # TODO: ensure startpoint is in the image.

        if self._check_projections_dir(overwrite=overwrite):
            return

        if not self.cached_trajectories_dir.exists():
            self.cached_trajectories_dir.mkdir()

        if not self.surface_data_dir.exists():
            self.surface_data_dir.mkdir()

        kwire_guide: Optional[List[tools.Tool]]

        # handle test_only option
        corridors_dir = self.dataset_dir / "corridors"
        if not corridors_dir.exists():
            corridors_dir.mkdir()

        corridor_surface_dir = self.dataset_dir / "corridor_surfaces"
        if not corridor_surface_dir.exists():
            corridor_surface_dir.mkdir()

        n = 0
        num_breached = 0
        total = self.total_images
        sample_dirs = []
        with Progress(refresh_per_second=1) as progressbar:
            task = progressbar.add_task(
                f"[red]Making kwire views dataset with {total} images...", total=total
            )
            for sample_idx, (ct_path, annotation_path) in enumerate(self.sample_paths):
                mode_idx = self.sample_modes[sample_idx]
                mode = self.mode_names[mode_idx]

                # Check whether the CT is in the test set and continue if appropriate.
                if self.test_only and mode != "test":
                    n += self.images_per_sample[mode_idx]
                    progressbar.advance(task, self.images_per_sample[mode_idx])
                    continue

                # Make directories corresponding to this sample.
                left_side = "left" in str(annotation_path)
                base = self.get_base_from_ct_path(ct_path) + ("_left" if left_side else "_right")
                sample_dir: Path = self.projections_dir / base
                if (
                    sample_dir.exists()
                    and len(self.get_images_in_sample_dir(sample_dir))
                    == self.images_per_sample[mode_idx]
                ):
                    n += self.images_per_sample[mode_idx]
                    progressbar.advance(task, self.images_per_sample[mode_idx])
                    sample_dirs.append(str(sample_dir))
                    continue
                elif not generate:
                    continue
                elif sample_dir.exists():
                    rmtree(sample_dir)

                # Going to make the sample, so add it to the list. (If it dies, no worries.)
                sample_dirs.append(str(sample_dir))
                sample_dir.mkdir()

                # Make or overwrite the trajectory dir for this sample.
                trajectory_dir: Path = self.cached_trajectories_dir / base
                if (
                    trajectory_dir.exists()
                    and len(list(trajectory_dir.glob("*.mrk.json"))) != self.num_trajectories
                ):
                    rmtree(trajectory_dir)
                if not trajectory_dir.exists():
                    trajectory_dir.mkdir(parents=True)

                fractured = "fractured" in annotation_path.name
                ct, segs, surfaces, trees = self.get_ct(
                    ct_path,
                    left_side,
                    imaging_center_in_world=self.device.isocenter_in_world,
                    use_material_segmentations="never",
                )

                log.debug(f"annotation_path: {annotation_path}")
                log.debug(f"ct_path: {ct_path}")
                annotation = self.load_annotations(annotation_path, ct)
                # Get the corridor
                left_side_str = "left" if left_side else "right"
                corridor_path = corridors_dir / f"{base}_corridor"
                corridor = self.get_corridor(
                    annotation,
                    segs[f"hip_{left_side_str}"],
                    trees[f"hip_{left_side_str}"],
                    trees[f"femur_{left_side_str}"],
                    corridor_path,
                )
                segs[f"corridor_{left_side_str}"] = corridor

                # Save the corridor surface
                corridor_surface_path = corridor_surface_dir / f"{base}_corridor.stl"
                if not corridor_surface_path.exists():
                    corridor_surface_LPS = corridor.isosurface(0.5, convert_to_LPS=True)
                    corridor_surface_LPS.save(corridor_surface_path)
                else:
                    corridor_surface_LPS = pv.read(corridor_surface_path)
                surfaces[f"corridor_{left_side_str}"] = corridor_surface_LPS.transform(
                    geo.get_data(ct.world_from_anatomical @ geo.RAS_from_LPS)
                )

                for t in range(self.num_trajectories):
                    # Reload the CT with material segmentations possibly used (or not, and bone density adjusted).
                    ct, _, _, _ = self.get_ct(
                        ct_path,
                        left_side,
                        world_from_anatomical=ct.world_from_anatomical,
                        use_material_segmentations="random",
                    )

                    # Scale the K-wire. Base model is 1mm in radius
                    kwire_diameter = np.random.uniform(1.5, 4)
                    kwire_density = np.random.uniform(2, 10)  # 30
                    kwire = deepdrr.vol.KWire.from_example(
                        diameter=kwire_diameter,
                        density=kwire_density,
                    )

                    # Adjusts the windowing level
                    intensity_upper_bound = np.random.uniform(3, 20)
                    self.projector_config["intensity_upper_bound"] = intensity_upper_bound

                    projector = deepdrr.Projector(
                        [ct, kwire],
                        device=self.device,
                        **self.projector_config,
                    )

                    seg_projectors: Dict[str, deepdrr.Projector] = {}
                    seg_projectors["kwire"] = deepdrr.Projector(
                        kwire,
                        device=self.device,
                        **self.seg_projector_config,
                    )
                    for name, seg in segs.items():
                        seg_projectors[name] = deepdrr.Projector(
                            seg,
                            device=self.device,
                            **self.seg_projector_config,
                        )

                    projector.initialize()
                    for name, seg_projector in seg_projectors.items():
                        seg_projector.initialize()

                    # Sample the trajectory.
                    trajectory_paths = sorted(list(trajectory_dir.glob("*.mrk.json")))
                    (
                        trajectory_annotation,
                        cortical_breach,
                        reason,
                        save_trajectory,
                    ) = self.get_trajectory(
                        trajectory_path=None,  # if t >= len(trajectory_paths) else trajectory_paths[t],
                        annotation=annotation,
                        tree=trees[f"hip_{left_side_str}"],
                        ct=ct,
                        seg=segs[f"hip_{left_side_str}"],
                        surface=surfaces[f"hip_{left_side_str}"],
                        fractured=fractured,
                    )
                    kwire_startpoint = trajectory_annotation.startpoint_in_world
                    kwire_endpoint = trajectory_annotation.endpoint_in_world
                    log.info(f"[{base}] cortical_breach: {cortical_breach}")

                    # Save the trajectory, depending
                    breach_str = "breach" if cortical_breach else "nobreach"
                    color = np.array([212, 4, 4] if cortical_breach else [24, 201, 8]) / 255.0
                    if save_trajectory:
                        trajectory_annotation.save(
                            trajectory_dir / f"traj-{t:06d}_{breach_str}_{reason}.mrk.json",
                            color=color,
                        )

                    point: geo.Point3D
                    ray: geo.Vector3D
                    for vidx, (point, ray) in enumerate(self.get_views(annotation, ct, mode)):

                        # Debugging
                        point = annotation.startpoint_in_world

                        # Move the Loop-X to the chosen view.
                        detector_to_point_distance = np.random.uniform(*self.detector_distance)
                        self.device.align(
                            point,
                            ray,
                            detector_to_point_distance=detector_to_point_distance,
                            detector_to_point_tolerance=50,
                            ray_angle_tolerance=0,
                            hover=True,
                            max_iters=100,
                        )

                        achieved_point = self.device.detector_slider_in_world
                        achieved_ray = self.device.principle_ray_in_world
                        point_diff = (achieved_point - point).norm()
                        ray_diff = achieved_ray.angle(ray)
                        if point_diff > 5 or ray_diff > math.radians(5):
                            log.warning(
                                f"Loop-X alignment error: {point_diff:.02f} mm, {math.degrees(ray_diff):.02f}{DEGREE_SIGN}"
                            )
                            log.debug(f"desired point, ray: {point}, {ray}")
                            log.debug(f"achieved point, ray: {achieved_point}, {achieved_ray}")

                        # log.info(f"{self.device}")

                        # Sample a progress value and move the kwire there.
                        progress = np.random.uniform(*self.progress_values)
                        kwire.align(kwire_startpoint, kwire_endpoint, progress)

                        # Get some other nice info to have.
                        camera3d_from_world = self.device.camera3d_from_world
                        index_from_world = self.device.get_camera_projection().index_from_world
                        principle_ray = self.device.principle_ray_in_world

                        # Now store everything that is needed to recreate this view.
                        info = dict(
                            ct=ct_path,
                            mode=mode,
                            anatomical_from_world=ct.anatomical_from_world,
                            world_from_anatomical=ct.world_from_anatomical,  # added 10/7/2022
                            anatomical_coordinate_system=ct.anatomical_coordinate_system,
                            annotation=annotation_path.name,
                            annotation_path=annotation_path,
                            base=base,
                            gt_startpoint=annotation.startpoint_in_world,
                            gt_endpoint=annotation.endpoint_in_world,
                            startpoint=kwire_startpoint,
                            endpoint=kwire_endpoint,
                            startpoint_in_anatomical=trajectory_annotation.startpoint,
                            endpoint_in_anatomical=trajectory_annotation.endpoint,
                            principle_ray=principle_ray,
                            progress=progress,
                            fractured=fractured,
                            cortical_breach=cortical_breach,
                            reason=reason,
                            device=self.device.get_config(),
                            index_from_world=index_from_world,
                            camera3d_from_world=camera3d_from_world,
                            source_to_detector_distance=self.device.source_to_detector_distance,
                            # Volume creation parameters
                            kwire_diameter=kwire_diameter,
                            kwire_density=kwire_density,
                            projector_config=self.projector_config,
                            view_point=achieved_point,
                            view_ray=achieved_ray,
                            datetime=datetime.now().isoformat(),
                        )

                        stem = f"t-{t:06d}_v-{vidx:06d}_{breach_str}"
                        instance_dir = sample_dir / stem
                        instance_dir.mkdir()
                        utils.save_json(instance_dir / f"{stem}_info.json", info)

                        # Image projection.
                        log.debug(
                            f"startpoint, endpoint in KWire anatomical: {kwire.anatomical_from_world @ annotation.startpoint_in_world}, {kwire.anatomical_from_world @ annotation.endpoint_in_world}"
                        )
                        image = projector()
                        image_utils.save(instance_dir / f"{stem}_drr.png", image)

                        for seg_name, seg_projector in seg_projectors.items():
                            seg_image = seg_projector()
                            # seg_image = np.where(seg_image > 0, 1, 0).astype(np.float32)
                            image_utils.save(instance_dir / f"{stem}_{seg_name}.png", seg_image)

                        # increment things
                        n += 1
                        num_breached += int(info["cortical_breach"])
                        progressbar.advance(task)

                projector.free()
                for seg_projector in seg_projectors.values():
                    seg_projector.free()

        log.info(
            f"{n} images contain {num_breached} instances of cortical breach, {n - num_breached} no breach."
        )
        return np.array(sample_dirs)

    """Methods to use during testing."""

    def analyze_image(
        self, model: UNet, image: np.ndarray, left_side: bool
    ) -> Tuple[np.ndarray, ...]:
        log.info(f"Running model on image...")
        image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(model.device)
        image_tensor = TF.resize(image_tensor, self.image_size)
        # left_side = np.array([not left_side], np.int32) # Remember the original images might have been flipped?
        left_side = np.array([left_side], np.int32)
        (
            _,
            kwire_mask_tensor,
            hip_mask_tensor,
            startpoint_heatmap_tensor,
            startpoint_depthmap_tensor,
            corridor_mask_tensor,
        ) = model.process_outputs(image_tensor, left_side, original_image_size=image.shape)
        kwire_mask = utils.get_numpy(kwire_mask_tensor)[0]
        hip_mask = utils.get_numpy(hip_mask_tensor)[0]
        startpoint_heatmap = utils.get_numpy(startpoint_heatmap_tensor)[0, 0]
        startpoint_depthmap = utils.get_numpy(startpoint_depthmap_tensor)[0, 0]
        corridor_mask = utils.get_numpy(corridor_mask_tensor)[0, 0]
        log.info(f"Done running model on image.")

        return kwire_mask, hip_mask, startpoint_heatmap, startpoint_depthmap, corridor_mask

    def save_images(self, base: str, **kwargs) -> Dict[str, Path]:
        """Save the images and info to the current images directory.

        TODO: Organize output images by "run," so that that corresponding 1st and 2nd images are in the same folder.

        """

        paths = {}

        for k, v in kwargs.items():
            p = f"images/{base}_{k}.png"
            image_utils.save(p, v)
            paths[k] = p
        return paths

    def triangulation(self, model: UNet):
        """Test triangulation using the given model on the whole dataset.

        Args:
            model: The model to test.
            xi: The xi value in radians for viewpoint planning.

        Returns:
            The test results.
        """

        model.eval()

        kwire = deepdrr.vol.KWire.from_example()

        results_file = open(f"results.csv", "w+")
        fieldnames = [
            "sample_idx",
            "base",
            "xi_deg",
            "Rotation Direction",
            "name",  # view name
            "failure",  # either empty or has a reason for failure
            "kwire_tip_error_mm",
            "kwire_angle_error_deg",
            "corridor_safe",  # comparison of corridor with surface
            "corridor_startpoint_error_mm",  # distance from gt startpoint to corridor line
            "corridor_angle_error_deg",  # relative angle
            "corridor_circumscribing_radius_mm",  # radius of the circumscribing cylinder from the annotation
            "cortical_breach_pred",  # from comparing predicted K-wire trajectory to predicted corridor
            "cortical_breach_gt",  # from comparing GT K-wire trajectory to surface
            "depth_error_mm",  # error at predicted startpoint
        ]
        csv_writer = csv.DictWriter(results_file, fieldnames=fieldnames)
        csv_writer.writeheader()
        sample_paths = self.sample_paths[-self.samples_per_mode[2] :]
        log.info(f"sample_paths:\n{sample_paths}")
        for sample_idx, (ct_path, annotation_path) in enumerate(
            track(sample_paths, description="Triangulate", total=len(sample_paths))
        ):
            # Make directories for this sample
            left_side = "left" in str(annotation_path)
            left_side_str = "left" if left_side else "right"
            log.debug(f"Loading {ct_path} and {annotation_path} for left_side={left_side}...")
            base = self.get_base_from_ct_path(ct_path) + (f"_left" if left_side else "_right")
            sample_dir = Path(base)
            sample_dir.mkdir()

            # Load the CT and annotations.
            ct, segs, surfaces, trees = self.get_ct(
                ct_path,
                left_side,
                imaging_center_in_world=self.device.isocenter_in_world,
                use_material_segmentations="never",
            )
            seg = segs[f"hip_{left_side_str}"]
            surface_in_world = surfaces[f"hip_{left_side_str}"]
            tree = trees[f"hip_{left_side_str}"]

            # Transform surface back, because efficiency!
            surface_lps: pv.PolyData = surface_in_world.copy()
            surface_lps.transform(ct.anatomical_from_world.get_data(), inplace=True)
            surface_lps.transform(geo.LPS_from_RAS.get_data(), inplace=True)
            surface_lps.save(sample_dir / "surface.stl")

            # Load the other hip, ONLY FOR VIS
            other_surface_in_world = surfaces[f"hip_{'left' if not left_side else 'right'}"]

            # Load the annotation
            annotation = self.load_annotations(annotation_path, ct)
            annotation.save(sample_dir / "gt_corridor.json", color=np.array([92, 184, 4]) / 255)
            projector = deepdrr.Projector([ct, kwire], device=self.device, **self.projector_config)
            projector.initialize()

            # Get the point
            for vidx, (name, point, ray) in enumerate(self.get_first_views(annotation, 5)):
                # for xi_degrees in [-10, -20, -30, -40]:
                for xi_degrees in [-30]:
                    # Negative for outlet, positive for inlet
                    xi = abs(np.radians(xi_degrees))
                    results = {}
                    results["sample_idx"] = sample_idx
                    results["base"] = base
                    results["name"] = name
                    results["xi_deg"] = abs(xi_degrees)
                    results["Rotation Direction"] = "Inlet" if xi_degrees > 0 else "Outlet"
                    corridor_dir = sample_dir / f"corridor-{xi_degrees}"
                    if not corridor_dir.exists():
                        corridor_dir.mkdir()

                    kwire_dir = sample_dir / f"kwire-{xi_degrees}"
                    if not kwire_dir.exists():
                        kwire_dir.mkdir()

                    # Sample a trajectory
                    kwire_startpoint, kwire_endpoint = self.sample_trajectory(
                        annotation, tree
                    )  # in world coordinates
                    kwire_annotation = LineAnnotation(
                        ct.anatomical_from_world @ kwire_startpoint,
                        ct.anatomical_from_world @ kwire_endpoint,
                        ct,
                    )
                    # kwire_annotation.save(
                    #     sample_dir / "gt_kwire.json", color=np.array([77, 102, 53]) / 255
                    # )
                    # Place the K-wire right on the bone.
                    kwire.align(
                        kwire_annotation.startpoint_in_world,
                        kwire_annotation.endpoint_in_world,
                        progress=0,
                    )

                    detector_to_point_distance = np.random.uniform(650, 700)
                    # TODO: maybe need to hover? Depends on where volume is.
                    # Instead of hovering, randomly adjust table height by moving volume.
                    self.device.align(
                        point,
                        # annotation.startpoint_in_world,
                        ray,
                        detector_to_point_distance=detector_to_point_distance,
                        detector_to_point_tolerance=100,
                        ray_angle_tolerance=0,
                        hover=True,
                    )

                    image_1 = projector()

                    # Run the model on the image.
                    (
                        kwire_mask_1,
                        hip_mask_1,
                        startpoint_heatmap_1,
                        startpoint_depthmap_1,
                        corridor_mask_1,
                    ) = self.analyze_image(model, image_1, not left_side)

                    image_base = f"{base}_{vidx:02d}_{name}"
                    # self.save_images(
                    #     image_base,
                    #     image_1=image_1,
                    #     kwire_mask_1=utils.combine_heatmap(image_1, kwire_mask_1 > 0.7),
                    #     # hip_mask_1=utils.combine_heatmap(image_1, hip_mask_1 > 0.5),
                    #     startpoint_heatmap_1=utils.combine_heatmap(image_1, startpoint_heatmap_1),
                    #     corridor_mask_1=utils.combine_heatmap(image_1, corridor_mask_1 > 0.5),
                    # )

                    (
                        corridor_line_in_index_1,
                        startpoint_in_index_1,
                        direction_in_index_1,
                    ) = eval_utils.locate_corridor(
                        startpoint_heatmap_1, corridor_mask_1, mask=hip_mask_1
                    )
                    if (
                        corridor_line_in_index_1 is None
                        or startpoint_in_index_1 is None
                        or direction_in_index_1 is None
                    ):
                        log.warning("Could not locate corridor in image 1.")
                        results["failure"] = "corridor_1_not_found"
                        csv_writer.writerow(results)
                        continue

                    # Sanity check on the startpoint
                    if (
                        startpoint_in_index_1[1] < 0
                        or startpoint_in_index_1[1] >= image_1.shape[0]
                        or startpoint_in_index_1[0] < 0
                        or startpoint_in_index_1[0] >= image_1.shape[1]
                    ):
                        log.warning("Startpoint 1 is out of bounds.")
                        results["failure"] = "startpoint_1_out_of_bounds"
                        csv_writer.writerow(results)
                        continue

                    # Estimate 3D startpoint
                    d = startpoint_depthmap_1[
                        int(startpoint_in_index_1[1]), int(startpoint_in_index_1[0])
                    ]
                    if d > 0.5:
                        log.warning(
                            f"Startpoint depth {d} is too large. Did you forget which model you're running?"
                        )
                    # Models trained with uncalibrated Loop-X (f = 1236.0000610351562)
                    # Real Loop-X has focal length of 1247.36, so add the difference when regressing depth.
                    # Current model has depth in range [-1, 1]
                    # Newer models should just multiply by the focal length (trained in range [0,1])
                    # Be sure to run this with the old_loopx urdf file.
                    # f_mm = 1369  # old loopx model
                    f_mm = 1247.36
                    # z_cam = (d / 2 + 0.5) * 1236.0000610351562 + (f_mm - 1236.0000610351562)
                    # z_cam = (d / 2 + 0.5) * 1236 + (f_mm - 1236)
                    z_cam = d * f_mm
                    log.debug(f"Startpoint depth {d} -> {z_cam}")
                    if z_cam < 400 or z_cam > 1200:
                        log.warning(f"Startpoint depth {z_cam} is out of range: {z_cam}.")
                        results["failure"] = "depth_out_of_range"
                        csv_writer.writerow(results)
                        continue

                    startpoint_ray_in_camera = geo.v(
                        self.device.camera3d_from_index
                        @ (
                            startpoint_in_index_1
                            + 50 * direction_in_index_1 * self.device.pixel_size
                        )
                    ).hat()
                    angle = startpoint_ray_in_camera.angle(geo.v(0, 0, 1))
                    desired_isocenter_in_camera = geo.p(0, 0, 0) + startpoint_ray_in_camera * (
                        z_cam / math.cos(angle)
                    )
                    desired_isocenter = (
                        self.device.world_from_camera3d @ desired_isocenter_in_camera
                    )

                    z_cam_gt = (self.device.camera3d_from_world @ annotation.startpoint_in_world).z
                    results["depth_error_mm"] = z_cam_gt - z_cam
                    log.debug(f"Depth error: {z_cam_gt - z_cam}")

                    # Run Preuhs algorithm. TODO: check rotation.
                    index_from_world_1 = self.device.index_from_world
                    device_from_preuhs = geo.FrameTransform.from_rt(
                        rotation=np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]]),
                        translation=self.device.device_from_world @ self.device.isocenter_in_world,
                    )
                    world_from_preuhs = self.device.world_from_device @ device_from_preuhs
                    index_from_preuhs = index_from_world_1 @ world_from_preuhs
                    c1 = geo.get_data(startpoint_in_index_1)
                    c2 = geo.get_data(startpoint_in_index_1 + 100 * direction_in_index_1)
                    P1 = geo.get_data(index_from_preuhs)

                    # Choose one of two ways to rotate.
                    ray_preuhs_1, _ = plan_viewpoint(c1, c2, P1, xi)
                    ray_preuhs_2, _ = plan_viewpoint(c1, c2, P1, -xi)
                    ray_world_1 = world_from_preuhs @ geo.Vector3D(ray_preuhs_1)
                    ray_world_2 = world_from_preuhs @ geo.Vector3D(ray_preuhs_2)
                    if xi_degrees > 0:
                        # Positive xi, rotate around X, so toward -Y
                        # For head-first, supine, this is toward inlet.
                        ray_world = ray_world_1 if ray_world_1.y < ray_world_2.y else ray_world_2
                    else:
                        ray_world = ray_world_1 if ray_world_1.y > ray_world_2.y else ray_world_2
                    # angle_1 = ray_world_1.angle(geo.v(0, 0, 1))
                    # angle_2 = ray_world_2.angle(geo.v(0, 0, 1))

                    # Re-align the device to the planned view.
                    self.device.align(
                        desired_isocenter,
                        ray_world,
                        ray_angle_tolerance=0,
                        detector_to_point_distance=600,  # if xi_degrees < 0 else 300,
                        detector_to_point_tolerance=100,
                        traction_yaw_tolerance=15,
                        min_gantry_tilt=-20,
                        max_gantry_tilt=20,
                        hover=True,
                    )
                    # TODO; check if alignment failed.
                    index_from_world_2 = self.device.index_from_world

                    # Get the second image
                    image_2 = projector()

                    # %%
                    # Visualization for viewpoint_planning figure
                    if abs(xi_degrees) >= 30:
                        pixel_size_at_plane = (
                            self.device.pixel_size / self.device.source_to_detector_distance * z_cam
                        )
                        cx_at_plane = pixel_size_at_plane * index_from_world_1.intrinsic.cx
                        cy_at_plane = pixel_size_at_plane * index_from_world_1.intrinsic.cy

                        f = geo.F(
                            np.array(
                                [
                                    [pixel_size_at_plane, 0, 0, -cx_at_plane],
                                    [0, pixel_size_at_plane, 0, -cy_at_plane],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1],
                                ]
                            )
                        )
                        startpoint_on_plane = (
                            index_from_world_1.world_from_camera3d
                            @ f
                            @ geo.p(startpoint_in_index_1.x, startpoint_in_index_1.y, z_cam)
                        )
                        endpoint_on_plane = (
                            index_from_world_1.world_from_camera3d
                            @ f
                            @ geo.p(
                                startpoint_in_index_1.x + 100 * direction_in_index_1.x,
                                startpoint_in_index_1.y + 100 * direction_in_index_1.y,
                                z_cam,
                            )
                        )
                        direction_on_plane = (endpoint_on_plane - startpoint_on_plane).hat()
                        axis_mesh = pv.Cylinder(startpoint_on_plane, direction_on_plane, 3, 400)
                        corridor_image_path_1 = image_utils.save(
                            f"corridors/{datetime.now():%Y-%m-%d_%H-%M-%S}_1.png",
                            utils.combine_heatmap(image_1, corridor_mask_1),
                        )
                        frustum_1, image_mesh_1 = vis.get_frustum_mesh(
                            index_from_world_1,
                            self.device.pixel_size,
                            image_path=corridor_image_path_1,
                            image_plane_distance=z_cam,
                            full_frustum=False,
                        )
                        frustum_2, image_mesh_2 = vis.get_frustum_mesh(
                            index_from_world_2,
                            self.device.pixel_size,
                            full_frustum=False,
                        )
                        plotter = pv.Plotter(window_size=(1536, 2048), off_screen=True)
                        # plotter = pv.Plotter(window_size=(1024, 1024), off_screen=True)
                        plotter.set_background("white")
                        plotter.add_mesh(
                            surface_in_world + other_surface_in_world, color="white", opacity=1
                        )
                        # cam_offset = geo.v(1800, 1800, 1300)
                        cam_offset = geo.v(-1000, 1000, 700)
                        cam_focus = (
                            index_from_world_1.center_in_world
                            + 300 * index_from_world_1.principle_ray_in_world
                        )
                        label_points = pv.PolyData(
                            np.array(
                                [
                                    index_from_world_1.center_in_world
                                    + 1000 * index_from_world_1.principle_ray_in_world,
                                    index_from_world_2.center_in_world
                                    + 1000 * index_from_world_2.principle_ray_in_world,
                                    startpoint_on_plane - direction_on_plane * 220,
                                ]
                            )
                        )
                        labels = ["Principle ray", "Planned view", "Axis of rotation"]
                        # plotter.add_point_labels(
                        #     label_points,
                        #     labels,
                        #     show_points=False,
                        #     font_size=24,
                        #     font_family="times",
                        #     text_color="black",
                        #     bold=False,
                        #     fill_shape=False,
                        #     shape_opacity=0.0,
                        # )
                        # plotter.add_axes()

                        plotter.add_mesh(axis_mesh, color="yellow", opacity=0.8)
                        plotter.set_position(index_from_world_2.center_in_world)
                        plotter.set_focus(
                            index_from_world_2.center_in_world
                            + self.device.source_to_detector_distance
                            * index_from_world_2.principle_ray_in_world
                        )
                        plotter.set_viewup(
                            list(index_from_world_2.world_from_camera3d @ geo.v(0, -1, 0))
                        )
                        image_utils.save(
                            f"screenshots/{sample_idx}_{name}_{xi_degrees}_{vidx}_view_2.png",
                            plotter.screenshot(return_img=True),
                        )
                        image_utils.save(
                            f"screenshots/{sample_idx}_{name}_{xi_degrees}_{vidx}_image_2.png",
                            image_2,
                        )

                        plotter.add_mesh(image_mesh_1, rgb=True, opacity=0.9)
                        plotter.set_position(index_from_world_1.center_in_world)
                        plotter.set_focus(
                            index_from_world_1.center_in_world
                            + self.device.source_to_detector_distance
                            * index_from_world_1.principle_ray_in_world
                        )
                        plotter.set_viewup(
                            list(index_from_world_1.world_from_camera3d @ geo.v(0, -1, 0))
                        )
                        image_utils.save(
                            f"screenshots/{sample_idx}_{name}_{xi_degrees}_{vidx}_view_1.png",
                            plotter.screenshot(return_img=True),
                        )
                        image_utils.save(
                            f"screenshots/{sample_idx}_{name}_{xi_degrees}_{vidx}_image_1.png",
                            image_1,
                        )

                        plotter.add_mesh(frustum_1, color="blue")
                        plotter.add_mesh(frustum_2, color="red")
                        plotter.set_position(cam_focus + cam_offset)
                        plotter.set_focus(cam_focus)
                        plotter.set_viewup(list(index_from_world_1.principle_ray_in_world))

                        # rot_axis = np.array(index_from_world_1.principle_ray_in_world)
                        # nframe = 200
                        # animations_dir = Path("animations")
                        # animations_dir.mkdir(exist_ok=True)
                        # animation_path = (
                        #     animations_dir
                        #     / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{sample_idx}_{name}_{xi_degrees}.gif"
                        # )
                        # plotter.open_gif(str(animation_path), loop=0, fps=30)
                        # log.info(f"Writing animation to {animation_path}")
                        # if abs(xi_degrees) >= 30:
                        #     for i, theta in enumerate(
                        #         np.pi / 3 * np.sin(np.linspace(0, 2 * np.pi, nframe, endpoint=True))
                        #     ):
                        #         if i % 10 == 0:
                        #             log.info(f"Writing frame {i} / {nframe}")

                        #         # Have the camera orbit around the principle ray
                        #         rot_mat = geo.f(geo.Rotation.from_rotvec(theta * rot_axis))
                        #         cam_pos = rot_mat @ cam_offset
                        #         plotter.set_position(cam_focus + cam_pos)
                        #         plotter.write_frame()
                        # plotter.close()

                        screenshot = plotter.screenshot(return_img=True)
                        image_utils.save(
                            f"screenshots/{sample_idx}_{name}_{xi_degrees}_{vidx}.png", screenshot
                        )

                    # %%

                    # Sanity check
                    gt_startpoint_in_index_2 = index_from_world_2 @ annotation.startpoint_in_world
                    if (
                        gt_startpoint_in_index_2[1] < 0
                        or gt_startpoint_in_index_2[1] >= image_2.shape[0]
                        or gt_startpoint_in_index_2[0] < 0
                        or gt_startpoint_in_index_2[0] >= image_2.shape[1]
                    ):
                        log.warning("Startpoint 2 out of bounds")
                        results["failure"] = "startpoint_2_out_of_bounds"
                        csv_writer.writerow(results)
                        continue

                    (
                        kwire_mask_2,
                        hip_mask_2,
                        startpoint_heatmap_2,
                        startpoint_depthmap_2,
                        corridor_mask_2,
                    ) = self.analyze_image(model, image_2, not left_side)
                    (
                        corridor_line_in_index_2,
                        startpoint_in_index_2,
                        direction_in_index_2,
                    ) = eval_utils.locate_corridor(
                        startpoint_heatmap_2, corridor_mask_2, mask=hip_mask_2
                    )

                    # Save out the second image results, for debugging
                    # self.save_images(
                    #     image_base,
                    #     image_2=image_2,
                    #     kwire_mask_2=utils.combine_heatmap(image_2, kwire_mask_2 > 0.7),
                    #     # hip_mask_2=utils.combine_heatmap(image_2, hip_mask_2 > 0.5),
                    #     startpoint_heatmap_2=utils.combine_heatmap(image_2, startpoint_heatmap_2),
                    #     corridor_mask_2=utils.combine_heatmap(image_2, corridor_mask_2 > 0.5),
                    # )

                    if (
                        corridor_line_in_index_2 is None
                        or startpoint_in_index_2 is None
                        or direction_in_index_2 is None
                    ):
                        log.warning("Corridor 2 not found")
                        results["failure"] = "corridor_2_not_found"
                        csv_writer.writerow(results)
                        continue

                    # Get the corridor in world space as the intersection of planes
                    corridor_plane_1 = corridor_line_in_index_1.backproject(index_from_world_1)
                    corridor_plane_2 = corridor_line_in_index_2.backproject(index_from_world_2)
                    corridor_in_world = corridor_plane_1.meet(corridor_plane_2)

                    # Get the approximate startpoint by intersecting the ray from the first view with the second plane.
                    startpoint_ray_1 = startpoint_in_index_1.backproject(index_from_world_1)
                    approx_startpoint_in_world = startpoint_ray_1.meet(corridor_plane_2)
                    startpoint_in_world = corridor_in_world.project(approx_startpoint_in_world)
                    direction_in_world = corridor_in_world.get_direction()

                    # ensure pointed in the right direction
                    projected_direction = (
                        index_from_world_2 @ (startpoint_in_world + direction_in_world)
                        - startpoint_in_index_2
                    )
                    if projected_direction.dot(direction_in_index_2) < 0:
                        direction_in_world = -direction_in_world

                    log.info(f"Adjusting corridor...")
                    startpoint_in_world, direction_in_world, radius = eval_utils.adjust_corridor(
                        corridor_heatmap_1=corridor_mask_1,
                        corridor_heatmap_2=corridor_mask_2,
                        hip_mask_1=hip_mask_1,
                        hip_mask_2=hip_mask_2,
                        index_from_world_1=index_from_world_1,
                        index_from_world_2=index_from_world_2,
                        startpoint_in_world=startpoint_in_world,
                        direction_in_world=direction_in_world,
                        radius=11,
                        image_1=image_1,
                        image_2=image_2,
                        t_bound=4,
                        angle_bound=np.radians(8),
                    )
                    corridor_in_world = geo.line(startpoint_in_world, direction_in_world)
                    log.info(f"Done adjusting corridor.")

                    # ensure pointed in the right direction (again)
                    projected_direction = (
                        index_from_world_2 @ (startpoint_in_world + 200 * direction_in_world)
                        - startpoint_in_index_2
                    )
                    if projected_direction.dot(direction_in_index_2) < 0:
                        direction_in_world = -direction_in_world

                    length = (annotation.endpoint_in_world - annotation.startpoint_in_world).norm()
                    endpoint_in_world = startpoint_in_world + 0.8 * length * direction_in_world
                    results[
                        "corridor_circumscribing_radius_mm"
                    ] = geo_utils.radius_of_circumscribing_cylinder(
                        annotation.startpoint_in_world,
                        annotation.startpoint_in_world.lerp(annotation.endpoint_in_world, 0.8),
                        corridor_in_world,
                    )

                    # Predict if the corridor is safe and save with the corresponding color
                    corridor_safe = False
                    # corridor_safe, _ = self.detect_cortical_breach(
                    #     ct,
                    #     seg,
                    #     surface_in_world,
                    #     tree,
                    #     annotation,
                    #     startpoint_in_world,
                    #     endpoint_in_world,
                    #     fractured="fractured" in annotation_path.name,
                    #     breach_detection_spacing=self.breach_detection_spacing,
                    #     breach_distance=self.breach_distance,
                    #     geodesic_threshold=self.geodesic_threshold,
                    #     cortical_breach_threshold=self.cortical_breach_threshold,
                    # )
                    corridor_annotation = LineAnnotation(
                        ct.anatomical_from_world @ startpoint_in_world,
                        ct.anatomical_from_world @ (startpoint_in_world + 200 * direction_in_world),
                        volume=ct,
                    )
                    corridor_annotation.save(
                        corridor_dir / f"corridor_{vidx:04d}_{xi_degrees}_{name}.mrk.json",
                        color=np.array([20, 173, 12] if corridor_safe else [176, 33, 14]) / 255,
                    )
                    results["corridor_safe"] = corridor_safe

                    corridor_startpoint_error_mm = corridor_in_world.distance(
                        annotation.startpoint_in_world
                    )
                    results["corridor_startpoint_error_mm"] = corridor_startpoint_error_mm
                    # angular error
                    corridor_angle_error_deg = np.degrees(
                        direction_in_world.angle(
                            annotation.endpoint_in_world - annotation.startpoint_in_world
                        )
                    )
                    if corridor_angle_error_deg > 90:
                        corridor_angle_error_deg = 180 - corridor_angle_error_deg
                    results["corridor_angle_error_deg"] = corridor_angle_error_deg

                    log.info(
                        f"Corridor error xi={xi_degrees}: {corridor_startpoint_error_mm:.2f} mm "
                        f"{corridor_angle_error_deg:.2f} deg"
                    )

                    # Now, find the K-wire in both images and figure out how far they are off.
                    (
                        kwire_line_in_index_1,
                        kwire_tip_in_index_1,
                        kwire_direction_in_index_1,
                    ) = eval_utils.locate_corridor(
                        startpoint_heatmap=None,
                        corridor_heatmap=kwire_mask_1,
                        startpoint=startpoint_in_index_1,
                        move_startpoint_to_mask=True,
                        threshold=0.7,
                    )
                    (
                        kwire_line_in_index_2,
                        kwire_tip_in_index_2,
                        kwire_direction_in_index_2,
                    ) = eval_utils.locate_corridor(
                        startpoint_heatmap=None,
                        corridor_heatmap=kwire_mask_2,
                        startpoint=startpoint_in_index_2,
                        move_startpoint_to_mask=True,
                        threshold=0.7,
                    )
                    if (
                        kwire_line_in_index_1 is None
                        or kwire_tip_in_index_1 is None
                        or kwire_direction_in_index_1 is None
                        or kwire_line_in_index_2 is None
                        or kwire_tip_in_index_2 is None
                        or kwire_direction_in_index_2 is None
                    ):
                        log.warning("K-wire not found")
                        results["failure"] = "kwire_not_found"
                    else:
                        kwire_plane_1 = kwire_line_in_index_1.backproject(index_from_world_1)
                        kwire_plane_2 = kwire_line_in_index_2.backproject(index_from_world_2)
                        kwire_in_world = kwire_plane_1.meet(kwire_plane_2)

                        # Get the tip location
                        kwire_tip_ray_1 = kwire_tip_in_index_1.backproject(index_from_world_1)
                        approx_kwire_tip_in_world = kwire_tip_ray_1.meet(kwire_plane_2)
                        kwire_tip_in_world = kwire_in_world.project(approx_kwire_tip_in_world)
                        kwire_direction_in_world = kwire_in_world.get_direction()

                        # ensure pointed in the right direction
                        kwire_projected_direction = (
                            index_from_world_2
                            @ (kwire_tip_in_world + 200 * kwire_direction_in_world)
                            - kwire_tip_in_index_2
                        )

                        # TODO: should be < 0, but for some reason it's > 0
                        if kwire_projected_direction.dot(kwire_direction_in_index_2) > 0:
                            kwire_direction_in_world = -kwire_direction_in_world

                        kwire_in_world = geo.line(kwire_tip_in_world, kwire_direction_in_world)

                        # record errors
                        results["kwire_tip_error_mm"] = kwire_in_world.distance(kwire.tip_in_world)
                        kwire_angle_error_deg = np.degrees(
                            kwire_direction_in_world.angle(kwire.tip_in_world - kwire.base_in_world)
                        )
                        if kwire_angle_error_deg > 90:
                            kwire_angle_error_deg = 180 - kwire_angle_error_deg
                        results["kwire_angle_error_deg"] = kwire_angle_error_deg

                        cortical_breach_pred = (
                            geo_utils.radius_of_circumscribing_cylinder(
                                kwire_tip_in_world,
                                kwire_tip_in_world + 90 * kwire_direction_in_world,
                                corridor_in_world,
                            )
                            > 3
                        )
                        results["cortical_breach_pred"] = cortical_breach_pred

                        pred_kwire_annotation = LineAnnotation(
                            ct.anatomical_from_world @ kwire_tip_in_world,
                            ct.anatomical_from_world
                            @ (kwire_tip_in_world + 200 * kwire_direction_in_world),
                            volume=ct,
                        )
                        pred_kwire_annotation.save(
                            kwire_dir / f"kwire_{vidx:04d}_{xi_degrees}_{name}.mrk.json",
                            color=np.array([20, 173, 12] if cortical_breach_pred else [176, 33, 14])
                            / 255,
                        )

                        cortical_breach_gt = True
                        # cortical_breach_gt, _ = self.detect_cortical_breach(
                        #     ct,
                        #     seg,
                        #     surface_in_world,
                        #     tree,
                        #     annotation,
                        #     kwire_startpoint,
                        #     kwire_endpoint,
                        #     fractured="fractured" in annotation_path.name,
                        #     breach_detection_spacing=self.breach_detection_spacing,
                        #     breach_distance=self.breach_distance,
                        #     geodesic_threshold=self.geodesic_threshold,
                        #     cortical_breach_threshold=self.cortical_breach_threshold,
                        # )
                        kwire_annotation.save(
                            kwire_dir / f"kwire_{vidx:04d}_{name}_gt.mrk.json",
                            color=np.array([27, 28, 27] if cortical_breach_gt else [217, 219, 217])
                            / 255,
                        )
                        results["cortical_breach_gt"] = cortical_breach_gt

                    # geo_utils.radius_of_circumscribing_cylinder(
                    #     startpoint_in_world,
                    #     endpoint_in_world,
                    # )

                    # TODO: rest of this, based onorridor_server. Should be easy. Don't make it hard.

                    log.debug(f"results: {results}")
                    csv_writer.writerow(results)

            projector.free()

        results_file.close()
        log.info("Done")


class CTPelvic1KDataModule(pl.LightningDataModule):
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
        CTPelvic1K(**self.dataset_kwargs)
        self.dataset_kwargs["download"] = False
        self.dataset_kwargs["generate"] = False

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_set = CTPelvic1K(mode="train", **self.dataset_kwargs)
            self.val_set = CTPelvic1K(mode="val", **self.dataset_kwargs)

        if stage == "test" or stage is None:
            self.test_set = CTPelvic1K(mode="test", **self.dataset_kwargs)

    def train_dataloader(
        self,
    ) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(self.train_set, shuffle=True, **self.loader_kwargs)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.val_set, **self.loader_kwargs)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.test_set, **self.loader_kwargs)
