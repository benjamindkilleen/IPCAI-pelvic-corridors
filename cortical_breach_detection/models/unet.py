import logging
from pathlib import Path
from re import S
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics
import torchmetrics.functional as TMF
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from cortical_breach_detection.losses.heatmap_linearity import HeatmapLinearityLoss
from cortical_breach_detection.tools.wire_guide import WireGuide
from deepdrr import geo
from deepdrr.annotations import LineAnnotation
from deepdrr.device import MobileCArm
from deepdrr.utils import image_utils
from matplotlib import pyplot as plt
from pytorch_lightning.callbacks import DeviceStatsMonitor
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import StochasticWeightAveraging
from scipy import optimize
from torch import nn
from torch import optim
from torch.nn import functional as F
from torchvision import models

from .. import tools
from .. import utils
from ..datasets.ctpelvic1k import CTPelvic1K
from ..datasets.ctpelvic1k import get_kwire_guide
from ..losses import DiceLoss2D
from ..losses import HeatmapLoss2D
from ..utils import nn_utils
from ..utils import eval_utils
from ..washer_detection import reconstruct_wire_and_guide

log = logging.getLogger(__name__)


DEBUG = False


class UNet(pl.LightningModule):
    """Basic U-Net module with dedicated channels for certain regressions.

    Based on:
    Paper: `U-Net: Convolutional Networks for Biomedical Image Segmentation
    <https://arxiv.org/abs/1505.04597>`_
    Paper authors: Olaf Ronneberger, Philipp Fischer, Thomas Brox
    Orinigally implemented by:
        - `Annika Brundyn <https://github.com/annikabrundyn>`_
        - `Akshay Kulkarni <https://github.com/akshaykvnit>`_

    Args:
        num_classes: Number of output channels required
        input_channels: Number of channels in input images (default 3)
        num_layers: Number of layers in each side of U-net (default 5)
        features_start: Number of features in first layer (default 64)
        bilinear: Whether to use bilinear interpolation or transposed convolutions (default) for upsampling.
    """

    kwire_guide: List[tools.Tool]

    def __init__(
        self,
        output_channels: int = 11,
        input_channels: int = 1,
        num_layers: int = 5,
        features_start: int = 64,
        bilinear: bool = False,
        optimizer: Dict[str, Any] = {"lr": 0.1, "momentum": 0.9, "weight_decay": 0.0005},
        scheduler: Dict[str, Any] = {"gamma": 0.1, "step_size": 45},
        kwire_guide: Optional[str] = "WireGuide",  # should match the dataset
        kwire_guide_density: float = 0.1,  # should match the dataset
        heatmap_loss_weight: float = 0.5,
        dice_loss_weight: float = 0.5,
        linearity_loss: bool = False,
        linearity_loss_weight: float = 0.1,
        linearity_loss_threshold: float = 0.5,
    ):
        if num_layers < 1:
            raise ValueError(f"num_layers = {num_layers}, expected: num_layers > 0")

        super().__init__()
        self.save_hyperparameters()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.kwire_guide = get_kwire_guide(kwire_guide, kwire_guide_density)
        self.heatmap_loss_weight = float(heatmap_loss_weight)
        self.dice_loss_weight = float(dice_loss_weight)
        self.linearity_loss_weight = float(linearity_loss_weight)

        # This shifts the range from [0,1] to [-1, 1]
        self.normalize = T.Normalize([0.5], [0.5])
        self.num_layers = num_layers
        self.output_channels = output_channels

        layers = [DoubleConv(input_channels, features_start)]

        feats = features_start
        for _ in range(num_layers - 1):
            layers.append(Down(feats, feats * 2))
            feats *= 2

        for _ in range(num_layers - 1):
            layers.append(Up(feats, feats // 2, bilinear))
            feats //= 2

        layers.append(nn.Conv2d(feats, self.output_channels, kernel_size=1))
        self.layers = nn.ModuleList(layers)

        self.mse_loss = nn.MSELoss()
        self.heatmap_loss = HeatmapLoss2D()
        self.dice_loss = DiceLoss2D(skip_bg=False)
        if linearity_loss:
            raise NotImplementedError("Linearity loss not implemented yet")
        else:
            self.linearity_loss = None

    def forward(self, x: torch.Tensor, original_image_size: Optional[Tuple[int, int]] = None):
        x = self.normalize(x)

        xi = [self.layers[0](x)]
        # Down path
        for layer in self.layers[1 : self.num_layers]:
            xi.append(layer(xi[-1]))

        # Up path
        for i, layer in enumerate(self.layers[self.num_layers : -1]):
            xi[-1] = layer(xi[-1], xi[-2 - i])
        y = self.layers[-1](xi[-1])

        if original_image_size is not None:
            y = F.interpolate(y, size=original_image_size, mode="bilinear", align_corners=False)

        # 0: kwire_mask
        # 1: hip_left_mask
        # 2: hip_right_mask
        # 3: femur_left_mask
        # 4: femur_right_mask
        # 5: sacrum_mask
        # 6: left_startpoint_heatmap
        # 7: right_startpoint_heatmap
        # 8: left_startpoint_depthmap
        # 9: right_startpoint_depthmap
        # 10: left_corridor_mask
        # 11: right_corridor_mask

        return dict(
            segs=y[:, 0:6, :, :],
            startpoint_heatmaps=y[:, 6:8, :, :],
            startpoint_depthmaps=y[:, 8:10, :, :],
            corridor_masks=y[:, 10:12, :, :],
        )

    def process_outputs(
        self,
        image: torch.Tensor,
        left_side: np.ndarray,
        original_image_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process the outputs.

        Args:
            image: Image tensor
            left_side (bool): (N,) Whether this is the patient's left side (True) or right side (False).

        Returns:

        """
        output = self(image, original_image_size)
        segs = output["segs"]  # kwire, left hip, right hip, left femur, right femur, sacrum
        startpoint_heatmap = output["startpoint_heatmaps"]  # left, right
        startpoint_depthmap = output["startpoint_depthmaps"]  # left, right
        corridor_mask = output["corridor_masks"]  # left, right

        # TODO: deal with size mismatch
        # if size is not None and heatmaps.shape[-2:] != size:
        #     heatmaps = TF.resize(heatmaps, size)
        #     segs = TF.resize(segs, size, interpolation=TF.InterpolationMode.NEAREST)
        #     image = TF.resize(image, size)

        # segs should be [background, guide, wire]
        # kwire_guide_mask = segs[:, 1] > 0.5
        # kwire_mask = segs[:, 2] > 0.5
        # kwire_mask = torch.logical_and(kwire_mask, torch.logical_not(kwire_guide_mask))
        kwire_mask = segs[:, 0]

        right_side = 1 - left_side

        startpoint_heatmap = torch.stack(
            [startpoint_heatmap[n, side_idx, :, :] for n, side_idx in enumerate(right_side)], dim=0
        )
        startpoint_depthmap = torch.stack(
            [startpoint_depthmap[n, side_idx, :, :] for n, side_idx in enumerate(right_side)],
            dim=0,
        )
        corridor_mask = torch.stack(
            [corridor_mask[n, side_idx, :, :] for n, side_idx in enumerate(right_side)], dim=0
        )
        hip_mask = torch.stack(
            [segs[n, side_idx + 1, :, :] for n, side_idx in enumerate(right_side)], dim=0
        )

        return (
            output,
            kwire_mask,
            hip_mask,
            startpoint_heatmap[:, None, :, :],
            startpoint_depthmap[:, None, :, :],
            corridor_mask[:, None, :, :],
        )

    def eval_step(self, batch, batch_idx, mode="val"):
        """Evaluate the model on a batch of data. Perform statistics depending on the mode."""
        image, target, info = batch
        training = mode == "train"

        # Get the right left/right heatmaps from the prediction
        left_side = utils.get_numpy(info["left_side"])
        (
            output,
            kwire_mask,
            hip_mask,
            startpoint_heatmap,
            startpoint_depthmap,
            corridor_masks,
        ) = self.process_outputs(image, left_side)

        loss = torch.tensor(0.0, device=image.device, dtype=image.dtype)

        # dice loss for kwire, left hip, right hip, and the corridor
        # TODO: consolidate segmentation losses. Dumb to have them separate
        # log.debug(f"segs: {output['segs'].shape}, target: {target['segs'].shape}")
        # log.debug(f"corridor: {corridor_masks.shape}, target: {target['corridor_masks'].shape}")

        # log.debug(f"kwire_mask: {kwire_mask.shape}, target: {target['segs'][:, 0].shape}")
        kwire_gt = target["segs"][:, 0:1, :, :]
        kwire_seg = output["segs"][:, 0:1, :, :]
        # kwire_gt = torch.logical_and(kwire_gt, image > 0.00784314)
        # kwire_seg = torch.logical_and(output["segs"][:, 0:1], image > 0.00784314)
        kwire_dice_loss = self.dice_loss(kwire_seg, kwire_gt)
        self.log(f"{mode}/kwire_dice_loss", kwire_dice_loss, prog_bar=training)

        dice_loss = self.dice_loss(output["segs"], target["segs"])
        corridor_dice_loss = self.dice_loss(corridor_masks, target["corridor_masks"])
        self.log(f"{mode}_corridor_dice_loss", corridor_dice_loss, prog_bar=True)
        dice_loss += corridor_dice_loss
        self.log(f"{mode}/dice_loss", dice_loss, prog_bar=True)
        loss += dice_loss

        # startpoint heatmap loss
        heatmap_loss = self.heatmap_loss(startpoint_heatmap, target["startpoint_heatmaps"])
        self.log(f"{mode}/startpoint_loss", heatmap_loss, prog_bar=training)
        loss += heatmap_loss
        # TODO: loss weighting

        # depthmap loss
        # TODO: figure out if it would matter if startpoint not in image
        d_error = torch.square(startpoint_depthmap - target["depth_maps"])
        d_idx = target["depth_maps"] > 0
        d_error_valid = d_error[d_idx]
        depth_loss = torch.mean(d_error_valid)
        self.log(f"{mode}/depth_loss", torch.std(d_error_valid), prog_bar=training)
        loss += depth_loss

        d_idx_flat = d_idx.any(dim=1).any(dim=1).any(dim=1)
        d = startpoint_depthmap[d_idx]
        # log.debug(f"d: {d.shape}")
        # log.debug(f"d_idx_flat: {d_idx_flat}, {d_idx_flat.sum()}")
        # log.debug("source_to_startpoint: {}".format(info["source_to_startpoint_mm"][d_idx_flat]))
        d_mm = d * info["source_to_detector_distance"][d_idx_flat]
        d_error_mm = torch.abs(d_mm - info["source_to_startpoint_mm"][d_idx_flat])
        d_mae_mm = torch.mean(d_error_mm)
        d_error_max_mm = torch.max(d_error_mm)
        self.log(f"{mode}/depth_mae_mm", d_mae_mm)
        self.log(f"{mode}/depth_max_error_mm", d_error_max_mm)

        loss /= 3

        self.log(f"loss/{mode}", loss)

        nan_loss = False
        if np.isnan(loss.detach().cpu().numpy()).any():
            log.error(
                f"Nan loss.\ndice_loss: {dice_loss}\nheatmap_loss: {heatmap_loss}\ndepth_loss: {depth_loss}\nloss: {loss}"
            )
            nan_loss = True

        save_images = (
            mode == "test"
            or (mode == "val" and batch_idx in [0, 1, 100, 101])
            or (mode == "train" and batch_idx in [0, 1, 100, 101])
            or nan_loss
        )
        run_reconstruction = (
            mode == "test" or (mode == "val" and batch_idx in [0, 1])
        ) and "startpoint" in info

        # initialize things to pass to the end of the epoch
        outputs = dict(loss=loss)
        startpoint_landmark_errors = []  # errors in mm of the startpoint in 2D in the image
        phi_errors = []  # error in radians of the angle of the safe corridor in the image
        front_guide_errors = []  # errors in mm of the front of the K-wire guide
        guide_angle_errors = []  # errors in radians of the angle of the k-wire

        num_tool_successes = 0  # Times the tool was successfully located
        num_tool_failures = 0  # Times the tool location failed, for whatever reason
        num_corridor_successes = 0
        num_corridor_failures = 0

        if save_images or run_reconstruction:
            images = utils.get_numpy(image)
            # kwire_guide_masks = utils.get_numpy(kwire_guide_mask)
            # binary_guide_masks = utils.get_numpy(kwire_guide_mask > 0.5)
            kwire_masks = utils.get_numpy(kwire_mask)
            binary_kwire_masks = utils.get_numpy(kwire_mask > 0.5)
            startpoint_heatmaps = utils.get_numpy(startpoint_heatmap[:, 0, :, :])
            corridor_masks = utils.get_numpy(corridor_masks[:, 0, :, :])

        if run_reconstruction:
            # Get info things
            startpoints = utils.get_numpy(info["startpoint"])
            endpoints = utils.get_numpy(info["endpoint"])
            gt_startpoints = utils.get_numpy(info["gt_startpoint"])
            gt_endpoints = utils.get_numpy(info["gt_endpoint"])
            index_from_worlds = utils.get_numpy(info["index_from_world"])
            world_from_indexes = utils.get_numpy(info["world_from_index"])
            camera3d_from_worlds = utils.get_numpy(info["camera3d_from_world"])
            startpoint_is_in_image = utils.get_numpy(info["startpoint_is_in_image"])
            # kwire_guide_progresses = utils.get_numpy(info["kwire_guide_progress"])
            source_to_detector_distances = utils.get_numpy(info["source_to_detector_distance"])
            pixel_sizes = utils.get_numpy(info["pixel_size"])
            phis = utils.get_numpy(info["phi"])
            ct_paths = info["ct_path"]
            cortical_breach_labels = utils.get_numpy(info["cortical_breach_label"])
            image_heights = utils.get_numpy(info["image_height"])
            image_widths = utils.get_numpy(info["image_width"])

            # Get output things
            for n in range(image.shape[0]):
                # Get basic things
                camera3d_from_world = geo.FrameTransform(camera3d_from_worlds[n])
                index_from_world = geo.Transform(index_from_worlds[n], _inv=world_from_indexes[n])
                gt_startpoint: geo.Point3D = geo.point(gt_startpoints[n])
                gt_endpoint: geo.Point3D = geo.point(gt_endpoints[n])
                startpoint = geo.point(startpoints[n])
                endpoint = geo.point(endpoints[n])
                # kwire_guide_progress = kwire_guide_progresses[n]
                gt_startpoint_in_index = index_from_world @ gt_startpoint

                # Convert from the resized image indices to the original image indices

                log.debug(
                    f"TODO: only take the corridor points inside the hip AND outside the femur?"
                )
                _, pred_startpoint_in_index, w_index = eval_utils.locate_corridor(
                    startpoint_heatmaps[n],
                    corridor_masks[n],
                    # original_image_size=(image_heights[n], image_widths[n]),
                )

                if pred_startpoint_in_index is None or w_index is None:
                    num_corridor_failures += 1
                else:
                    landmark_error = (pred_startpoint_in_index - gt_startpoint_in_index).norm()
                    landmark_error_mm = pixel_sizes[n] * landmark_error

                    gt_w_index: geo.Vector3D = geo.vector(
                        (index_from_world @ gt_endpoint) - (index_from_world @ gt_startpoint)
                    ).hat()
                    phi_error = w_index.angle(gt_w_index)
                    log.debug(f"phi_error: {np.degrees(phi_error):.02f} degrees")
                    phi_errors.append(phi_error)

                    startpoint_landmark_errors.append(landmark_error_mm)
                    log.debug(
                        f"landmark: {pred_startpoint_in_index}, gt: {gt_startpoint_in_index}, error: {landmark_error:.02f} pixels, {landmark_error_mm:.02f} mm"
                    )
                    num_corridor_successes += 1

                # Get the error on the K-wire
                log.debug(f"TODO: segment the K-wire and get the error on that angle.")

        metrics_dir = Path(f"{mode}_{self.current_epoch:03d}_metrics")
        if not metrics_dir.exists():
            metrics_dir.mkdir()
        utils.write_metrics(
            metrics_dir / "startpoint_landmark_error.txt", startpoint_landmark_errors
        )
        utils.write_metrics(metrics_dir / "phi_error.txt", phi_errors)
        utils.write_metrics(metrics_dir / "num_corridor_successes.txt", [num_corridor_successes])
        utils.write_metrics(metrics_dir / "num_corridor_failures.txt", [num_corridor_failures])
        if self.kwire_guide is not None:
            utils.write_metrics(metrics_dir / "front_guide_error.txt", front_guide_errors)
            utils.write_metrics(metrics_dir / "guide_angle_error.txt", guide_angle_errors)
            utils.write_metrics(metrics_dir / "num_tool_successes.txt", [num_tool_successes])
            utils.write_metrics(metrics_dir / "num_tool_failures.txt", [num_tool_failures])
        utils.write_metrics(metrics_dir / "sample_base.txt", info["base"])

        outputs["startpoint_landmark_error"] = torch.tensor(startpoint_landmark_errors)
        outputs["phi_error"] = torch.tensor(phi_errors)
        outputs["num_corridor_successes"] = torch.tensor(num_corridor_successes)
        outputs["num_corridor_failures"] = torch.tensor(num_corridor_failures)

        if self.kwire_guide is not None:
            outputs["front_guide_error"] = torch.tensor(front_guide_errors)
            outputs["guide_angle_error"] = torch.tensor(guide_angle_errors)
            outputs["num_tool_successes"] = torch.tensor(num_tool_successes)
            outputs["num_tool_failures"] = torch.tensor(num_tool_failures)

        if save_images:
            for n in range(image.shape[0]):
                sample_base = info["base"][n]
                sample_dir = Path(f"{mode}_{self.current_epoch:03d}_{sample_base}")
                if not sample_dir.exists():
                    sample_dir.mkdir()

                # Save the images
                image_path = Path(info["image_path"][n])
                image_base = image_path.stem[:-4]
                image_utils.save(sample_dir / f"{image_base}_image.png", images[n, 0])
                image_utils.save(
                    sample_dir / f"{image_base}_startpoint-heatmap.png",
                    utils.combine_heatmap(image[n], startpoint_heatmap[n]),
                )
                image_utils.save(
                    sample_dir / f"{image_base}_startpoint-heatmap_gt.png",
                    utils.combine_heatmap(image[n], target["startpoint_heatmaps"][n, 0]),
                )
                image_utils.save(
                    sample_dir / f"{image_base}_corridor-mask.png",
                    utils.combine_heatmap(image[n], corridor_masks[n] > 0.5),
                )
                image_utils.save(
                    sample_dir / f"{image_base}_corridor-mask_gt.png",
                    utils.combine_heatmap(image[n], target["corridor_masks"][n, 0] > 0.5),
                )

                # Save the kwire mask images
                image_utils.save(
                    sample_dir / f"{image_base}_wire-mask_gt.png",
                    utils.combine_heatmap(image[n], kwire_gt[n, 0] > 0.5),
                )
                image_utils.save(
                    sample_dir / f"{image_base}_wire-mask.png",
                    utils.combine_heatmap(image[n], kwire_seg[n, 0] > 0.5),
                )

                # Save the hip mask images
                image_utils.save(
                    sample_dir / f"{image_base}_hip_mask.png",
                    utils.combine_heatmap(image[n], hip_mask[n] > 0.5),
                )
                image_utils.save(
                    sample_dir / f"{image_base}_left-hip_mask_gt.png",
                    utils.combine_heatmap(image[n], target["segs"][n, 1] > 0),
                )
                image_utils.save(
                    sample_dir / f"{image_base}_right-hip_mask_gt.png",
                    utils.combine_heatmap(image[n], target["segs"][n, 2] > 0),
                )
                if training:
                    break

        if nan_loss and training:
            raise ValueError("NaN loss")

        return outputs

    def training_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, mode="train")["loss"]

    def eval_epoch_end(self, outputs, mode="val"):
        startpoint_landmark_error = utils.get_numpy(
            torch.cat([o["startpoint_landmark_error"] for o in outputs])
        )
        phi_error = utils.get_numpy(torch.cat([o["phi_error"] for o in outputs]))
        num_corridor_successes = sum(utils.get_numpy(o["num_corridor_successes"]) for o in outputs)
        num_corridor_failures = sum(utils.get_numpy(o["num_corridor_failures"]) for o in outputs)
        corridor_success_rate = num_corridor_successes / (
            num_corridor_successes + num_corridor_failures
        )

        if self.kwire_guide is not None:
            front_guide_error = utils.get_numpy(
                torch.cat([o["front_guide_error"] for o in outputs])
            )
            guide_angle_error = utils.get_numpy(
                torch.cat([o["guide_angle_error"] for o in outputs])
            )
            num_tool_successes = sum(utils.get_numpy(o["num_tool_successes"]) for o in outputs)
            num_tool_failures = sum(utils.get_numpy(o["num_tool_failures"]) for o in outputs)
            tool_success_rate = num_tool_successes / (num_tool_successes + num_tool_failures)

        log.info(
            f"startpoint landmark error: {startpoint_landmark_error.mean():.02f} +/- {startpoint_landmark_error.std():.02f} mm"
        )
        log.info(
            f"phi error: {np.degrees(phi_error.mean()):.02f} +/- {np.degrees(phi_error.std()):.02f} degrees"
        )
        log.info(
            f"corridor successes: {num_corridor_successes} / {num_corridor_successes + num_corridor_failures} ({100 * corridor_success_rate:.01f}%)"
        )

        if self.kwire_guide is not None:
            log.info(
                f"front guide error: {front_guide_error.mean():.02f} +/- {front_guide_error.std():.02f} mm"
            )
            log.info(
                f"guide angle error: {np.degrees(guide_angle_error.mean()):.02f} +/- {np.degrees(guide_angle_error.std()):.02f} degrees"
            )
            log.info(
                f"tool successes: {num_tool_successes} / {num_tool_successes + num_tool_failures} ({100 * tool_success_rate:.01f}%)"
            )

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        return self.eval_epoch_end(outputs)

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "test")

    def test_epoch_end(self, outputs):
        return self.eval_epoch_end(outputs, "test")

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), **self.optimizer)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, **self.scheduler)
        return dict(optimizer=optimizer, lr_scheduler=lr_scheduler, monitor="loss/train")

    def configure_callbacks(self):
        return [
            StochasticWeightAveraging(),
            LearningRateMonitor(logging_interval="step"),
            DeviceStatsMonitor(),
            ModelCheckpoint(save_last=True, every_n_train_steps=1),
        ]

    def analyze_image(self, image: np.ndarray, left_side: bool, image_size: Tuple[int, int]):
        log.info(f"Running model on image...")
        image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(self.device)
        image_tensor = TF.resize(image_tensor, image_size)
        # left_side = np.array(
        #     [not left_side], np.int32
        # )  # Remember the original images might have been flipped?
        left_side = np.array([left_side], np.int32)
        (
            _,
            kwire_mask_tensor,
            hip_mask_tensor,
            startpoint_heatmap_tensor,
            startpoint_depthmap_tensor,
            corridor_mask_tensor,
        ) = self.process_outputs(image_tensor, left_side, original_image_size=image.shape)
        kwire_mask = utils.get_numpy(kwire_mask_tensor)[0]
        hip_mask = utils.get_numpy(hip_mask_tensor)[0]
        startpoint_heatmap = utils.get_numpy(startpoint_heatmap_tensor)[0, 0]
        startpoint_depthmap = utils.get_numpy(startpoint_depthmap_tensor)[0, 0]
        corridor_mask = utils.get_numpy(corridor_mask_tensor)[0, 0]
        log.info(f"Done running model on image.")

        return kwire_mask, hip_mask, startpoint_heatmap, startpoint_depthmap, corridor_mask

    def triangulate(
        self,
        image_1: np.ndarray,
        image_2: np.ndarray,
        index_from_world_1: geo.CameraProjection,
        index_from_world_2: geo.CameraProjection,
        left_side: bool,
        image_size: Tuple[int, int],
    ) -> Tuple[
        Optional[geo.Point3D], Optional[geo.Vector3D], Optional[geo.Point3D], Optional[geo.Vector3D]
    ]:
        """Triangulate the corridor in the two images.

        Args:
            image_1: The first image.
            image_2: The second image.
            index_from_world_1: The camera projection for the first image.
            index_from_world_2: The camera projection for the second image.
            image_size: The size at which to evaluate the images.

        """

        # Run the model on the image.
        (
            kwire_mask_1,
            hip_mask_1,
            startpoint_heatmap_1,
            startpoint_depthmap_1,
            corridor_mask_1,
        ) = self.analyze_image(image_1, left_side, image_size)

        (
            corridor_line_in_index_1,
            startpoint_in_index_1,
            direction_in_index_1,
        ) = eval_utils.locate_corridor(startpoint_heatmap_1, corridor_mask_1, mask=hip_mask_1)
        if (
            corridor_line_in_index_1 is None
            or startpoint_in_index_1 is None
            or direction_in_index_1 is None
        ):
            return None, None, None, None

        # Sanity check on the startpoint
        if (
            startpoint_in_index_1[1] < 0
            or startpoint_in_index_1[1] >= image_1.shape[0]
            or startpoint_in_index_1[0] < 0
            or startpoint_in_index_1[0] >= image_1.shape[1]
        ):
            return None, None, None, None

        (
            kwire_mask_2,
            hip_mask_2,
            startpoint_heatmap_2,
            startpoint_depthmap_2,
            corridor_mask_2,
        ) = self.analyze_image(image_2, left_side, image_size)
        (
            corridor_line_in_index_2,
            startpoint_in_index_2,
            direction_in_index_2,
        ) = eval_utils.locate_corridor(startpoint_heatmap_2, corridor_mask_2, mask=hip_mask_2)

        if (
            corridor_line_in_index_2 is None
            or startpoint_in_index_2 is None
            or direction_in_index_2 is None
        ):
            return None, None, None, None

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
            index_from_world_2 @ (startpoint_in_world + 200 * direction_in_world)
            - startpoint_in_index_2
        )
        if projected_direction.dot(direction_in_index_2) < 0:
            log.debug("Flipping direction before adjustment")
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
            threshold=0.01,
        )
        corridor_in_world = geo.line(startpoint_in_world, direction_in_world)
        log.info(f"Done adjusting corridor.")

        # ensure pointed in the right direction (again)
        projected_direction = (
            index_from_world_2 @ (startpoint_in_world + 200 * direction_in_world)
            - startpoint_in_index_2
        )
        if projected_direction.dot(direction_in_index_2) < 0:
            log.debug(f"Reversing direction after adjustment.")
            direction_in_world = -direction_in_world

        # TODO: triangulate K-wire as well if it's in the image
        kwire_tip_in_world = None
        kwire_direction_in_world = None

        return startpoint_in_world, direction_in_world, kwire_tip_in_world, kwire_direction_in_world


class DoubleConv(nn.Module):
    """
    [ Conv2d => BatchNorm (optional) => ReLU ] x 2
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """
    Downscale with MaxPool => DoubleConvolution block
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """
    Upsampling (by either bilinear interpolation or transpose convolutions)
    followed by concatenation of feature map from contracting path, followed by DoubleConv.
    """

    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = False):
        super().__init__()
        self.upsample = None
        if bilinear:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_ch, in_ch // 2, kernel_size=1),
            )
        else:
            self.upsample = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)

        # Pad x1 to the size of x2
        diff_h = x2.shape[2] - x1.shape[2]
        diff_w = x2.shape[3] - x1.shape[3]

        x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])

        # Concatenate along the channels axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
