from typing import Tuple, Union
import numpy as np

import imgaug.augmenters as iaa


def gaussian_adjust_contrast(
    images: np.ndarray,
    alpha: Union[float, Tuple[float, float]] = (0.6, 1.4),
    sigma: Union[float, Tuple[float, float]] = (0.1, 0.5),
):
    N, H, W, C = images.shape
    if isinstance(alpha, tuple):
        alpha = np.random.uniform(alpha[0], alpha[1])
    if isinstance(sigma, tuple):
        s = np.random.uniform(sigma[0], sigma[1]) * min(H, W)
    else:
        s = sigma * min(H, W)

    mu_x = np.random.uniform(0, H, size=N)
    mu_y = np.random.uniform(0, W, size=N)
    xs, ys = np.meshgrid(
        np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing="ij"
    )
    xdiff = xs[:, :, None] - mu_x[None, None, :]
    ydiff = ys[:, :, None] - mu_y[None, None, :]
    distance_squared = xdiff**2 + ydiff**2
    h = np.exp(-distance_squared / (2 * s * s))
    hmax = np.max(h, axis=(0, 1), keepdims=True)
    hmap = h / hmax  # in [0, 1]
    alpha_map = hmap * (alpha - 1) + 1
    images = 0.5 + (images - 0.5) * alpha_map
    return np.clip(images, 0, 1).astype(np.float32)


def gaussian_contrast(
    alpha: Union[float, Tuple[float, float]] = (0.6, 1.4),
    sigma: Union[float, Tuple[float, float]] = (0.1, 0.5),
):
    """Nonuniform contrast augmentation.

    Adjust the contrast by scaling each pixel with value `v` at `x` to
    `0.5 + (v - 0.5) * exp(-(x - mu)**2 / (2 * sigma**2)))`

    Args:
        alpha (float or tuple of float): Alpha of the nonuniform contrast
            augmentation. If a tuple is provided, the value will be randomly
            selected from the range.
        sigma (float or tuple of float): Standard deviation of the Gaussian
            kernel, as a fraction of the (smaller) image size. If a tuple is provided, the value
            will be randomly selected from the range.
    """
    if isinstance(alpha, tuple):
        assert len(alpha) == 2
        assert alpha[0] <= alpha[1]

    if isinstance(sigma, tuple):
        assert len(sigma) == 2
        assert sigma[0] <= sigma[1]
        sigma = np.random.uniform(sigma[0], sigma[1])

    def func_images(images, random_state, parents, hooks):
        # Images are in NHWC
        return gaussian_adjust_contrast(images, alpha, sigma)

    return iaa.Lambda(func_images=func_images)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from PIL import Image
    from deepdrr.utils import image_utils

    fig, (ax1, ax2) = plt.subplots(2, 1)
    image = (
        np.array(
            Image.open(
                "/home/killeen/projects/cortical-breach-detection/datasets/CTPelvic1K/projections/dataset6_CLINIC_0102_right/t-000000_v-000000_breach_drr.png"
            )
        ).astype(np.float32)
        / 255
    )
    images = image[np.newaxis, :, :, np.newaxis]
    aug_images = gaussian_adjust_contrast(images, 1.9, 0.4)
    aug_image = aug_images[0, :, :, 0]
    image_utils.save("images/contrast_image_orig.png", image)
    image_utils.save("images/contrast_image_aug.png", aug_image)
