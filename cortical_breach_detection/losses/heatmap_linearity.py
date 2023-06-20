import logging
from typing import Optional
import torch
from torch import nn
import numpy as np

from ..utils import nn_utils

log = logging.getLogger(__name__)


def heatmap_linearity_loss(
    heatmap: torch.Tensor, threshold: float = 0.5, max_points: int = 100, min_points: int = 10
) -> torch.Tensor:
    """
    Compute the linearity loss of a single heatmap.
    """
    # log.debug(f"heatmap: {heatmap.min()} {heatmap.max()}")
    threshold = nn_utils.get_heatmap_threshold(heatmap, fraction=threshold)
    # heatmap = nn_utils.normalize_heatmap(heatmap, min_range=0.1)
    H, W = heatmap.shape
    # log.debug(f"normalized heatmap: {heatmap.min()} {heatmap.max()}")
    # TODO:
    # log.debug(f"threshold: {threshold}")
    rr, cc = torch.where(heatmap > threshold)
    num_points = rr.shape[0]
    # log.debug(f"num_points: {num_points}")

    if num_points < min_points:
        return torch.tensor(0.0)
    elif num_points > max_points:
        # TODO: take random subset for speed.
        # log.debug(f"Too many points in heatmap: {points_index.shape[0]}, taking random subset.")
        indexing = np.random.choice(rr.shape[0], size=100, replace=False)
        rr = rr[indexing]
        cc = cc[indexing]

    points_index = torch.stack((rr, cc), dim=1)
    # Direction of line is the principle component of variation
    normalized_heatmap = nn_utils.normalize_heatmap(heatmap)
    # log.debug(f"normalized_heatmap: {normalized_heatmap.min()} {normalized_heatmap.max()}")
    heatmap_values = normalized_heatmap[points_index[:, 0], points_index[:, 1]]
    c = torch.sum(heatmap_values[:, None] * points_index, dim=0) / (
        torch.sum(heatmap_values) + 1e-8
    )
    # log.debug(f"c: {c}")
    _, _, vv_T = torch.linalg.svd(points_index.to(torch.float32) - c[None, :].to(torch.float32))
    vv_T.to(heatmap.dtype)

    # Principle comonents are the columns of V
    v = vv_T[0]

    # homogeneous line connecting the two points is their cross product
    c = torch.cat([c, torch.ones_like(c[:1])], dim=0)
    v = torch.cat([v, torch.zeros_like(v[:1])], dim=0)
    l = torch.cross(c, c + v)
    # log.debug(f"l: {l}")

    # Homogeneous points. (Use all points above threshold, not just the random subset.)
    ps = torch.stack((rr, cc, torch.ones_like(rr)), dim=1).to(heatmap.dtype)
    D_sqr = (ps @ l[:, None]).square() / (l[:2].square().sum() + 1e-8)
    # log.debug(f"D_sqr: {D_sqr.min()} {D_sqr.max()}")
    h_D_sqr = heatmap_values * D_sqr
    # log.debug(f"h_D_sqr: {h_D_sqr.min()} {h_D_sqr.max()}")
    rmse = torch.sqrt(h_D_sqr.mean())
    return rmse


class HeatmapLinearityLoss(nn.Module):
    """Loss function that penalizes heatmaps with low linearity."""

    def __init__(self, threshold: float = 0.5):
        """Loss function that penalizes heatmaps with low linearity.

        Args:
            threshold (float, optional): Threshold at which to run fit on the data. Defaults to 0.5.
            sigma (float, optional): Expected width of the heatmap. Defaults to 1.0.
        """

        super(HeatmapLinearityLoss, self).__init__()
        self.threshold = threshold

    def forward(self, heatmap: torch.Tensor):
        return sum(
            [heatmap_linearity_loss(h, self.threshold) for h in heatmap],
            torch.tensor(0, dtype=heatmap.dtype, device=heatmap.device),
        ).mean()
