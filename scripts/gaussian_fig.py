import matplotlib.pyplot as plt
import numpy as np
from cortical_breach_detection.utils.nn_utils import heatmap_numpy
from deepdrr.utils import image_utils


def main(image_size: int = 256):

    # Generate some data
    h = heatmap_numpy(
        0.75 * image_size, 0.25 * image_size, 0.25 * image_size, (image_size, image_size)
    )
    cmap = plt.get_cmap("viridis")

    # Map h to an image
    Z = np.zeros((image_size, image_size, 3))
    for i in range(image_size):
        for j in range(image_size):
            Z[i, j, :] = cmap(h[i, j])[:3]

    image_utils.save("images/gaussian.png", Z)


if __name__ == "__main__":
    main()
