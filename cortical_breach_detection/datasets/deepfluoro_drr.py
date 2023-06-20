from __future__ import absolute_import
from __future__ import annotations

import itertools
import json
import logging
import re
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

import deepdrr
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.transforms.functional as TF
from deepdrr import geo, Volume
from deepdrr.utils import image_utils
from PIL import Image
from rich.progress import BarColumn
from rich.progress import Progress
from rich.progress import TimeRemainingColumn
from scipy.signal import argrelmax
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.utils import download_url, download_and_extract_archive
from deepdrr.load_dicom import (
    conv_hu_to_density,
    conv_hu_to_materials_thresholding,
    conv_hu_to_materials,
)
from .ctpelvic1k import CTPelvic1K

from .. import utils
from ..console import console

log = logging.getLogger(__name__)

specimen_groups = ["17-1882", "17-1905", "18-0725", "18-1109", "18-2799", "18-2800"]


class DeepFluoroSpecimen:
    def __init__(self, data, specimen_group: Union[str, int]):
        self.data = data
        self.specimen_group = specimen_group
        # if isinstance(specimen_group, int):
        #     self.specimen_group = specimen_groups[specimen_group]
        # else:
        assert (
            self.specimen_group in specimen_groups
        ), f"unrecognized specimen group: {self.specimen_group}"

    @staticmethod
    def _segment_materials(
        hu_values: np.ndarray,
        use_thresholding: bool = True,
        cache_dir: Optional[Path] = None,
        use_cached: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Segment the materials.
        Meant for internal use, particularly to be overriden by volumes with different materials.
        Args:
            hu_values (np.ndarray): volume data in Hounsfield Units.
            use_thretholding (bool, optional): whether to segment with thresholding (true) or a DNN. Defaults to True.
        Returns:
            Dict[str, np.ndarray]: materials segmentation.
        """

        if use_thresholding:
            return conv_hu_to_materials_thresholding(hu_values)
        else:
            return conv_hu_to_materials(hu_values)

    def get_volume(
        self,
        use_thresholding: bool = True,
        cache_dir: Optional[Path] = None,
        use_cached: bool = True,
    ):

        # Reading HU values of the volumn
        hu_volume = np.array(self.data[self.specimen_group]["vol"]["pixels"]).astype(np.float32)
        hu_volume = np.moveaxis(hu_volume, [0, 1, 2], [2, 1, 0]).copy()

        # convert HU values to density
        data = conv_hu_to_density(hu_volume)

        # segment materials
        materials = self._segment_materials(hu_volume, use_thresholding=use_thresholding)

        # Reading in the origin of the volume, and voxel spacing in world-space
        origin = np.array(self.data[self.specimen_group]["vol"]["origin"]).squeeze()
        spacing = np.array(self.data[self.specimen_group]["vol"]["spacing"]).squeeze()
        rotation_mat = np.array(self.data[self.specimen_group]["vol"]["dir-mat"])

        anatomical_from_ijk = geo.FrameTransform.from_rt(rotation=rotation_mat, translation=origin)
        RAS_from_ijk = geo.RAS_from_LPS @ anatomical_from_ijk
        world_from_anatomical = geo.LPS_from_RAS

        return Volume(
            data=data,
            materials=materials,
            anatomical_from_ijk=RAS_from_ijk,
            world_from_anatomical=world_from_anatomical,
            anatomical_coordinate_system="RAS",
        )


class DeepFluoro(CTPelvic1K):
    def __init__(
        self,
        root: str = "~/datasets",
        mode="test",
        download: bool = False,
        generate: bool = False,
        overwrite: bool = False,
        carm: Dict[str, Any] = {},
        projector: Dict[str, Any] = {},
    ):
        """Make the dataset.
        Args:
            root (str, optional): Where datasets are stored in general. Defaults to "~/datasets".
            download: Download the CT or X-Ray data.
            max_startpoint_offset (float, optional): How far to offset the startpoint in mm. Defaults to 8.
            max_endpoint_offset (float, optional): How far to offset the endpoint in mm. Defaults to 15.
            num_trajectories_per_sample (int, optional): Number of trajectories to randomly sample.
                The first trajectory is always the annotated one.
                Additional trajectories are randomly sampled by moving the start and endpoints. Defaults to 10.
            num_points_per_trajectory (int, optional): Number of points along the trajectory to
                place the KWire at, spaced regularly from a random start point near the entry to the endpoint. Defaults to 10.
            num_views_per_point (int, optional): Number of views to capture for each point on each trajectory,
                for each sample. The views cycle through AP, obturator oblique, and iliac oblique,
                plus some randomness. Defaults to 1.
            cortical_breach_threshold (float, optional): The density value (not the HU value!) above which peaks
                are considered a cortical breach. See `conv_hu_to_density` for the conversion.
                Defaults to 1.45.
            num_breach_detection_points: Number of points in the CT to interpolate along the trajectory for detecting the breach.
            carm (Dict[str, Any], optional): Keyword arguments to pass to the MobileCArm. Defaults to {}.
            projector (Dict[str, Any], optional): Keyword arguments to pass to the projector. Defaults to {}.
        """
        self.root = Path(root).expanduser()
        self.dataset_dir = self.root / "Deepfluoro"
        self.mode = mode
        assert self.mode == "test"
        self.max_startpoint_offset = max_startpoint_offset
        self.max_endpoint_offset = max_endpoint_offset
        self.num_trajectories_per_view = num_trajectories_per_view
        self.progress_values = np.array(progress_values)
        self.num_points_per_trajectory = len(self.progress_values)
        self.num_views_per_sample = num_views_per_sample
        self.cortical_breach_threshold = cortical_breach_threshold
        self.num_breach_detection_points = num_breach_detection_points
        self.max_alpha_error = max_alpha_error
        self.max_beta_error = max_beta_error
        self.max_isocenter_error = max_isocenter_error
        self.split = split
        self.carm = deepdrr.MobileCArm(**carm)
        self.projector_config = projector

        self.clinic_data_dir = self.dataset_dir / "ipcai_2020_full_res_data.h5"
        self.clinic_trajectories_dir = self.dataset_dir / "trajectories"
        self.kwire_views_dir = self.dataset_dir / "kwire_views"
        self.images_per_sample = (
            self.num_views_per_sample
            * self.num_points_per_trajectory
            * self.num_trajectories_per_view
        )

        self.sample_paths = self._get_sample_paths()
        if generate:
            self.generate(overwrite)

        self.image_paths = self.kwire_views_dir.glob("*/*/*.png")

    @property
    def data_folder(self):
        return self.root

    @property
    def data_path(self):
        return self.clinic_data_dir

    @property
    def data(self):
        return h5py.File(self.data_path, "r")

    def _get_sample_paths(self):
        sample_paths = []
        for specimen_group in sorted(list(specimen_groups)):
            annotation_paths = self.clinic_trajectories_dir.glob(f"img_{specimen_group}*.mrk.json")
            annotation_paths = sorted(list(annotation_paths))
            for annotation_path in annotation_paths:
                sample_paths.append((specimen_group, annotation_path))
        return sample_paths

    def get_ct(self, identifier):
        volume = DeepFluoroSpecimen(self.data, identifier).get_volume()
        return volume, identifier

    def download(self, url):
        if self._check_exists():
            return
        self.data_folder.mkdir(exist_ok=True)
        download_and_extract_archive(url, self.data_folder, remove_finished=True)

    def get_base_from_ct_path(self, ct_path: Path) -> Optional[str]:
        return ""


class DeepFluoroDataModule(pl.LightningDataModule):
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
        DeepFluoro(**self.dataset_kwargs)
        self.dataset_kwargs["download"] = False
        self.dataset_kwargs["generate"] = False

    def setup(self):
        self.test_set = DeepFluoro(mode="test", **self.dataset_kwargs)
        self.dims = self.test_set[0][0].shape

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.test_set, **self.loader_kwargs)
