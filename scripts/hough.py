"""Test the hough transform on an example image.

```
python scripts/hough.py
```

Uses files in teh "images" folder.

"""
import logging
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from rich.logging import RichHandler
from skimage.feature import canny
from skimage.transform import hough_line

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, handlers=[RichHandler()])


def hough_transform(x: np.ndarray, theta_resolution=np.pi / 180) -> np.ndarray:
    """Take the hough transform of the provided image.

    Args:
        x (np.ndarray): A [H,W] binary image.

    Returns:
        np.ndarray: Hough space.
    """

    angles = np.linspace(-np.pi / 2, np.pi / 2, theta_resolution, endpoint=False)
    hspace, theta, d = hough_line(x, angles)
    return


def main():
    image = cv2.imread("images/0001_obturator-oblique_oop_nobreach.png")
    gray = image[:, :, 0]
    edges = cv2.Canny(gray, 80, 140, apertureSize=3)
    # edges = canny(gray, sigma=3).astype(np.uint8) * 255
    cv2.imwrite("images/edges.png", edges)

    angles = np.linspace(-np.pi / 2, np.pi / 2, 720, endpoint=False)
    hspace, theta, d = hough_line(edges, angles)
    angle_step = 0.5 * np.diff(theta).mean()
    d_step = 0.5 * np.diff(d).mean()
    bounds = [np.rad2deg(theta[0] - angle_step), np.rad2deg(theta[-1] + angle_step), d[-1] + d_step, d[0] - d_step]

    plt.imshow(np.log(hspace + 1), extent=bounds, cmap="magma", aspect="auto")
    plt.title("Hough Transform")
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Distance (pixels)")
    plt.colorbar()
    plt.savefig("images/hspace.png")

    # lines = cv2.HoughLines(edges, 0.5, np.pi/720, 50)
    # if lines is None:
    #     raise RuntimeError("no lines found")
    # log.debug(lines.shape)

    # for rho,theta in lines[:, 0]:
    #     a = np.cos(theta)
    #     b = np.sin(theta)
    #     x0 = a*rho
    #     y0 = b*rho
    #     x1 = int(x0 + 1000*(-b))
    #     y1 = int(y0 + 1000*(a))
    #     x2 = int(x0 - 1000*(-b))
    #     y2 = int(y0 - 1000*(a))

    #     cv2.line(image,(x1,y1),(x2,y2),(0,0,255),1)

    # rhos, thetas = lines[:, 0, 0], lines[:, 0, 1]
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # ax1.plot(np.zeros_like(rhos), rhos, 'b.')
    # ax1.set_title("Rhos")
    # ax2.plot(np.zeros_like(thetas), thetas, 'r.')
    # ax2.set_title("Thetas")
    # plt.plot(rhos, np.degrees(thetas), 'b.')
    # plt.xlabel("Rho")
    # plt.ylabel("Theta")
    # plt.savefig("images/hough_params.png")

    # rho, theta = np.median(lines[:, 0], 0)
    # # rho, theta = np.mean(lines[:, 0], 0)
    # a = np.cos(theta)
    # b = np.sin(theta)
    # x0 = a*rho
    # y0 = b*rho
    # x1 = int(x0 + 1000*(-b))
    # y1 = int(y0 + 1000*(a))
    # x2 = int(x0 - 1000*(-b))
    # y2 = int(y0 - 1000*(a))

    # cv2.line(image,(x1,y1),(x2,y2),(0,0,255),1)
    # cv2.imwrite('images/hough.png', image)

    # HoughP
    # minLineLength = 30
    # maxLineGap = 5
    # lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength, maxLineGap)
    # log.debug(lines.shape)
    # for x1, y1, x2, y2 in lines[:, 0]:
    #     m = (y2 - y1) / (x2 - x1)
    #     x1_new = 0
    #     x2_new = image.shape[0]
    #     y1_new = round(y1 + m * (x1_new - x1))
    #     y2_new = round(y1 + m * (x2_new - x1))

    #     log.debug(f"stuff: {x1, y1, x2, y2}")
    #     log.debug(f"stuff: {x1_new, y1_new, x2_new, y2_new}")

    #     # cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), thickness=1)
    #     cv2.line(image, (x1_new, y1_new), (x2_new, y2_new), (0, 255, 0), thickness=1)

    # cv2.imwrite("images/houghP.png", image)


if __name__ == "__main__":
    main()
