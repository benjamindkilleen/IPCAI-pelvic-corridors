from typing import Union

import deepdrr
import matplotlib as mpl
import numpy as np
from deepdrr.utils import image_utils

mpl.use("agg")
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["font.family"] = "Times New Roman"

from PIL import Image


class LUT:
    def __init__(self, xs: np.ndarray, ys: np.ndarray):
        """Look-up table.

        Maps a point x to the linear interpolation between the points on either side of it.

        Args:
            xs (np.ndarray): [n] Array of domain values. Must be sorted. Anything outside this range is constant.
            ys (np.ndarray): [n - 1, 2] Array of corresponding range values, such that if x is in [xs[i], xs[i+1]], it is mapped to ys[i, 0], ys[i, 1].
        """
        self.xs = np.array(xs)
        self.ys = np.array(ys)

        assert self.xs.ndim == 1 and self.ys.ndim == 2 and self.ys.shape[1] == 2
        assert self.ys.shape[0] == self.xs.shape[0] - 1
        if not np.all(self.xs[:-1] <= self.xs[1:]):
            raise RuntimeError(f"xs array not sorted: {xs}")

    def __call__(self, x: Union[float, np.ndarray]):
        """Map the value(s) to its corresponding range.

        Args:
            x (np.ndarray): A single vallue or array of values to map.
        """

        shape = x.shape
        values = np.atleast_1d(x).reshape(-1)
        out = np.empty_like(values)
        for i, v in enumerate(values):
            if v < self.xs[0]:
                out[i] = self.ys[0, 0]
                continue

            if v >= self.xs[-1]:
                out[i] = self.ys[-1, 1]
                continue

            for j, x0 in enumerate(self.xs[:-1]):
                x1 = self.xs[j + 1]
                if v >= x0 and v < x1:
                    y0, y1 = self.ys[j]
                    t = (v - x0) / (x1 - x0)
                    out[i] = (1 - t) * y0 + t * y1
                    break

        return out.reshape(*shape)


def main():
    image_path = "/home/benjamin/datasets/2022-02_Bayview_Cadaver/BayviewExp_Feb25/Xraytiff_ori/36.tiff"
    out_path = "images/test_output.png"

    image = np.array(Image.open(image_path))
    border = 20  # 10
    image = image[border : image.shape[0] - 2 * border, border : image.shape[1] - border]
    # image[:border] = image_max
    # image[image.shape[0] - 2 * border :] = image_max
    # image[:, :border] = image_max
    # image[:, image.shape[1] - border :] = image_max

    sns.histplot(image.flat, stat="density")
    plt.savefig("images/hist_real.png", dpi=150)
    plt.close()

    # lut = LUT(
    #     [0, 10000, 27500, 45000, 48000, 52000],
    #     [[24000, 25000], [25000, 30000], [30000, 40000], [40000, 45000], [45000, 47000]],
    # )
    # image = lut(image)
    # image = np.clip(image, 25000, 47000)
    image = deepdrr.utils.neglog(image)

    drr_mean, drr_std = 0.3949881, 0.12929274
    image_mean, image_std = image.mean(), image.std()

    image = (image - image_mean) / image_std * drr_std + drr_mean
    print(f"real mean, std: {image.mean(), image.std()}")
    image = np.clip(image, 0, 1)

    sns.histplot(image.flat, stat="density")
    plt.savefig("images/hist_real_neglog.png", dpi=150)
    plt.close()
    image_utils.save("images/real_remapped.png", image)

    sns.histplot(image.flat, stat="density")
    plt.savefig("images/hist_real_remapped.png", dpi=150)
    plt.close()

    drr_path = "/home/benjamin/datasets/CTPelvic1K/projections_06-views_12-trajs_-5-00-05-10-15_oop_WireGuide/dataset6_CLINIC_0029_left/000_anteroposterior/p--05_t-00_000_breach.png"
    drr = np.array(Image.open(drr_path)).astype(np.float32) / 255
    sns.histplot(drr.flat, stat="density")
    plt.savefig("images/hist_drr.png", dpi=150)
    plt.close()
    print(f"drr mean, std: {drr.mean(), drr.std()}")

    image_utils.save("images/drr.png", drr)


if __name__ == "__main__":
    main()
