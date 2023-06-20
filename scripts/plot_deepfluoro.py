import logging
from pathlib import Path

import matplotlib as mpl
import numpy as np
from rich.logging import RichHandler

mpl.use("agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import click

log = logging.getLogger(__name__)
logging.basicConfig(handlers=[RichHandler()])
log.setLevel(logging.DEBUG)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.figsize"] = (4, 3)

DEGREES_SIGN = "\N{DEGREE SIGN}"


@click.command()
@click.option(
    "--results-dir",
    type=str,
    default="/home/killeen/projects/cortical-breach-detection/results/deepfluoro/2022-11-13_14-03-04_thor",
    help="Path to the deepfluoro results.",
)
@click.option("--linewidth", type=float, default=1, help="Width of the lines in the plot.")
def main(results_dir: str, linewidth: float):
    results_dir = Path(results_dir)
    images_dir = Path("images")

    data = pd.read_csv(results_dir / "results.csv")
    data["rotation_about_corridor_bins"] = pd.cut(
        data["rotation_about_corridor_deg"], bins=np.arange(10, 45, 5)
    )
    log.debug(data["rotation_about_corridor_bins"].unique())
    data["rotation_about_corridor_mids"] = [b.left for b in data["rotation_about_corridor_bins"]]

    kwargs = dict(linewidth=linewidth, fliersize=2, palette="crest")

    for xi, upper in zip([15, 20, 25, 30], [20, 25, 30, 50]):
        indices = np.logical_and(
            xi < data["rotation_about_corridor_deg"], data["rotation_about_corridor_deg"] <= upper
        )
        log.debug(f"{xi}: {indices.sum()} / {len(indices)}")
        t_error = data.loc[indices, "corridor_startpoint_error_mm"].mean()
        theta_error = data.loc[indices, "corridor_angle_error_deg"].mean()
        t_std = data.loc[indices, "corridor_startpoint_error_mm"].std()
        theta_std = data.loc[indices, "corridor_angle_error_deg"].std()
        log.info(
            f"error: {t_error:.1f} $\\pm$ {t_std:.1f} mm and {theta_error:.1f} $\\pm$ {theta_std:.1f}\\textdegree"
        )

    image_keys = set(data["image_1"].unique())
    image_keys.update(data["image_2"].unique())
    log.info(f"Found {len(image_keys)} images..")

    sns.boxplot(
        data,
        x="rotation_about_corridor_mids",
        y="corridor_startpoint_error_mm",
        **kwargs,
    )
    plt.tight_layout(pad=1.3)
    plt.xlabel(f"Relative Angle ({DEGREES_SIGN})")
    plt.ylabel("Translation Error (mm)")
    plt.ylim(0, 20)
    plt.savefig(images_dir / "deepfluoro_corridor_startpoint_error.png", dpi=300)
    plt.close()

    sns.boxplot(
        data,
        x="rotation_about_corridor_mids",
        y="corridor_angle_error_deg",
        **kwargs,
    )
    plt.tight_layout(pad=1.3)
    plt.xlabel(f"Relative Angle ({DEGREES_SIGN})")
    plt.ylabel(f"Rotation Error ({DEGREES_SIGN})")
    plt.ylim(0, 20)
    plt.savefig(images_dir / "deepfluoro_corridor_angle_error.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
