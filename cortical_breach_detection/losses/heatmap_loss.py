# 2D normalized cross correlation, operates on BxRxC arrays
# where B is batch, R is rows, C is columns. Outputs B length
# 1D array of NCC scores for each batch
#
# Copyright (C) 2019-2020 Robert Grupp (grupp@jhu.edu)
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
import logging

import torch
from torch import nn

log = logging.getLogger(__name__)


def ncc_2d(X, Y):
    N = X.shape[-1] * X.shape[-2]
    assert N > 1

    # print('X: {}'.format(X.shape))
    # print('Y: {}'.format(Y.shape))

    dim = X.dim()
    d1 = dim - 2
    d2 = dim - 1

    # compute means of each 2D "image"
    mu_X = torch.mean(X, dim=[d1, d2])

    # make the 2D images have zero mean
    X_zm = X - (mu_X.reshape(*mu_X.shape, 1, 1) * torch.ones_like(X))

    # compute sample standard deviations
    X_sd = torch.sqrt(torch.sum(X_zm * X_zm, dim=[d1, d2]) / (N - 1))

    mu_Y = torch.mean(Y, dim=[d1, d2])

    Y_zm = Y - (mu_Y.reshape(*mu_Y.shape, 1, 1) * torch.ones_like(Y))

    Y_sd = torch.sqrt(torch.sum(Y_zm * Y_zm, dim=[d1, d2]) / (N - 1))

    return torch.sum(X_zm * Y_zm, dim=[d1, d2]) / ((N * (X_sd * Y_sd)) + 1.0e-8)


class HeatmapLoss2D(nn.Module):
    def __init__(self):
        super(HeatmapLoss2D, self).__init__()

    def forward(self, inputs: torch.Tensor, target: torch.Tensor):
        num_landmarks = target.shape[1]

        ncc_losses = ncc_2d(inputs, target)

        # negation since we are minmizing, normalize output in range [-1,0]
        # ncc_losses = (ncc_losses + 1) * -0.5

        # negation since we are minmizing, normalize output in range [0,1]
        ncc_losses = (1 - ncc_losses) / 2

        return torch.mean(ncc_losses)
