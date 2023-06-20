#!/usr/bin/env python3
import logging
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import deepdrr
import numpy as np
import pyvista as pv
from deepdrr import geo
from stringcase import camelcase
from stringcase import snakecase

from .. import utils

log = logging.getLogger(__name__)


class Tool(deepdrr.Volume, ABC):
    """A class for representing tools based on voxelized surface models."""

    # Every tool should define the tip in anatomical coordinates, which is the center point begin inserted into the body along
    # the main axis of the tool, and the base, another point on that axis, so that they can be aligned.
    base: geo.Point3D
    tip: geo.Point3D
    radius: float

    # Available materials may be found at:
    # https://www.nist.gov/pml/x-ray-mass-attenuation-coefficients

    _material_mapping = {
        "ABS Plastic": "polyethylene",
        "Ceramic": "concrete",
        "Stainless Steel": "iron",
        "stainless_steel": "iron",
        "cement": "concrete",
        "plastic": "polyethylene",
        "metal": "iron",
        "bone": "bone",
        "titanium": "titanium",
    }

    _default_densities = {
        "polyethylene": 1.05,  # polyethyelene is 0.97, but ABS plastic is 1.05
        "concrete": 1.5,
        "iron": 7.5,
        "titanium": 7,
        "bone": 1.5,
    }

    NUM_POINTS = 4000

    def __init__(
        self,
        density: float = 0.1,
        world_from_anatomical: Optional[geo.FrameTransform] = None,
        densities: Dict[str, float] = {},
    ):
        """Create the tool.

        Args:
            density: The spacing of the voxelization for each component of the tool.
            world_from_anatomical: Defines the pose of the tool in world.
            densities: Optional overrides to the material densities

        """
        self.density = density
        self._densities = self._default_densities.copy()
        self._densities.update(densities)

        self.surfaces = {}
        bounds = []
        for material_dir, model_paths in self.get_model_paths():
            surface = pv.PolyData()
            for p in model_paths:
                s = pv.read(p)
                if len(s.points) > self.NUM_POINTS:
                    s = s.decimate(1 - self.NUM_POINTS / len(s.points))
                surface += s

            material_dirname = material_dir.name if isinstance(material_dir, Path) else material_dir
            self.surfaces[material_dirname] = surface
            bounds.append(surface.bounds)

        bounds = np.array(bounds)
        x_min, y_min, z_min = bounds[:, [0, 2, 4]].min(0)
        x_max, y_max, z_max = bounds[:, [1, 3, 5]].max(0)
        bounds = [x_min, x_max, y_min, y_max, z_min, z_max]

        cache_dir = self.get_cache_dir()
        materials_path = cache_dir / "materials.npz".format()
        anatomical_from_ijk_path = cache_dir / "anatomical_from_ijk.npy"
        if materials_path.exists() and anatomical_from_ijk_path.exists():
            log.debug(f"using cached voxelization: {materials_path.absolute()}")
            materials = dict(np.load(materials_path))
            anatomical_from_ijk = geo.FrameTransform(np.load(anatomical_from_ijk_path))
        else:

            materials, anatomical_from_ijk = self._get_materials(density, bounds)
            np.savez(materials_path, **materials)
            np.save(anatomical_from_ijk_path, geo.get_data(anatomical_from_ijk))

        # Convert from actual materials to DeepDRR compatible.
        materials = dict((self._material_mapping[m], seg) for m, seg in materials.items())

        data = np.zeros_like(list(materials.values())[0], dtype=np.float64)
        for material, seg in materials.items():
            data += self._densities[material] * seg

        super().__init__(
            data,
            materials,
            anatomical_from_ijk,
            world_from_anatomical,
            anatomical_coordinate_system=None,
        )

    @abstractmethod
    def get_model_paths(self) -> List[Tuple[Path, List[Path]]]:

        pass

    def get_model_paths(self) -> List[Tuple[Path, List[Path]]]:
        """Get the model paths associated with this Tool.

        By default looks in data_dir / snakecase(classname)

        Returns:
            List[Tuple[Path, List[Path]]]: List of tuples containing the material dir and a list of paths with STL files for that material.
        """
        data_dir = utils.get_drawings_dir()
        stl_dir = data_dir / snakecase(camelcase(self.__class__.__name__))
        model_paths = [(p.stem, [p]) for p in stl_dir.glob("*.stl")]
        if not model_paths:
            raise FileNotFoundError(
                f"couldn't find materials for {self.__class__.__name__} in {stl_dir}"
            )
        return model_paths

    def get_cache_dir(self) -> Path:
        cache_dir = (
            utils.get_cache_dir()
            / self.__class__.__name__
            / "{}mm".format(str(self.density).replace(".", "-"))
        )
        cache_dir.mkdir(exist_ok=True, parents=True)
        return cache_dir

    def _get_materials(self, density, bounds):
        materials = {}
        for material, surface in self.surfaces.items():
            log.info(f'voxelizing {self.__class__.__name__} "{material}" (may take a while)...')
            materials[material], anatomical_from_ijk = utils.surface_utils.voxelize(
                surface,
                density=density,
                bounds=bounds,
            )

        return materials, anatomical_from_ijk

    @property
    def base_in_world(self) -> geo.Point3D:
        return self.world_from_anatomical @ self.base

    @property
    def tip_in_world(self) -> geo.Point3D:
        return self.world_from_anatomical @ self.tip

    @property
    def length_in_world(self):
        return (self.tip_in_world - self.base_in_world).norm()

    def align(self, startpoint: geo.Point3D, endpoint: geo.Point3D, progress: float = 1):
        """Place the tool along the line between startpoint and endpoint.

        Args:
            startpoint (geo.Point3D): Startpoint in world.
            endpoint (geo.Point3D): Point in world toward which the cannula points.
            progress (float): The fraction between startpoint and endpoint to place the tip of the cannula. Defaults to 1.
        """
        # useful: https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d

        # interpolate along the direction of the tool to get the desired points in world.
        startpoint = geo.point(startpoint)
        endpoint = geo.point(endpoint)
        progress = float(progress)
        trajectory_vector = endpoint - startpoint

        desired_tip_in_world = endpoint - (1 - progress) * trajectory_vector
        desired_base_in_world = (
            desired_tip_in_world - trajectory_vector.hat() * self.length_in_world
        )

        self.world_from_anatomical = geo.FrameTransform.from_line_segments(
            desired_tip_in_world,
            desired_base_in_world,
            self.tip,
            self.base,
        )

    def twist(self, angle: float, degrees: bool = True):
        """Rotate the tool clockwise (when looking down on it) by `angle`.

        Args:
            angle (float): The angle.
            degrees (bool, optional): Whether `angle` is in degrees. Defaults to True.
        """
        rotvec = (self.tip - self.base).hat()
        rotvec *= deepdrr.utils.radians(angle, degrees=degrees)
        self.world_from_anatomical = self.world_from_anatomical @ geo.frame_transform(
            geo.Rotation.from_rotvec(rotvec)
        )

    def get_mesh_in_world(self, full: bool = True, use_cached: bool = True):
        mesh = sum(self.surfaces.values(), pv.PolyData())
        mesh.transform(geo.get_data(self.world_from_anatomical), inplace=True)
        # meshh += pv.Sphere(
        #     center=list(self.world_from_ijk @ geo.point(0, 0, 0)), radius=5
        # )

        x, y, z = np.array(self.shape) - 1
        points = [
            [0, 0, 0],
            [0, 0, z],
            [0, y, 0],
            [0, y, z],
            [x, 0, 0],
            [x, 0, z],
            [x, y, 0],
            [x, y, z],
        ]

        points = [list(self.world_from_ijk @ geo.point(p)) for p in points]
        mesh += pv.Line(points[0], points[1])
        mesh += pv.Line(points[0], points[2])
        mesh += pv.Line(points[3], points[1])
        mesh += pv.Line(points[3], points[2])
        mesh += pv.Line(points[4], points[5])
        mesh += pv.Line(points[4], points[6])
        mesh += pv.Line(points[7], points[5])
        mesh += pv.Line(points[7], points[6])
        mesh += pv.Line(points[0], points[4])
        mesh += pv.Line(points[1], points[5])
        mesh += pv.Line(points[2], points[6])
        mesh += pv.Line(points[3], points[7])

        return mesh

    @property
    def center(self) -> geo.Point3D:
        return self.base.lerp(self.tip, 0.5)

    @property
    def center_in_world(self) -> geo.Point3D:
        return self.world_from_anatomical @ self.center
