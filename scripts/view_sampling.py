from typing import List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from deepdrr import geo


def sample_spherical(d_phi: float, n: int) -> np.ndarray:
    """Sample n vectors within `phi` radians of [0, 0, 1]."""
    theta = np.random.uniform(0, 2 * np.pi, n)

    phi = np.arccos(np.random.uniform(np.cos(d_phi), 1, n))

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    return np.stack([x, y, z], axis=1)


def spherical_uniform(v: geo.Vector3D, d_phi: float, n: int = 1) -> List[geo.Vector3D]:
    """Sample unit vectors within `d_phi` radians of `v`."""
    v = geo.vector(v).hat()
    points = sample_spherical(d_phi, n)
    F = v.rotation(geo.vector(0, 0, 1))
    return [F @ geo.vector(p) for p in points]


def main():
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    points = np.array(spherical_uniform([0, 0, 1], np.pi, 100))
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])

    points = np.array(spherical_uniform([1, 1, 1], np.pi / 6, 500))
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])

    plt.show()


if __name__ == "__main__":
    main()
