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
    default="/home/killeen/projects/cortical-breach-detection/results/triangulate/2022-11-15_10-33-36_thor_test-set_outlet",
    help="Path to the triangulation results.",
)
@click.option("--linewidth", type=float, default=1, help="Width of the lines in the plot.")
def main(results_dir: str, linewidth: float):
    results_dir = Path(results_dir)
    images_dir = Path("images")
    log.debug(results_dir)

    data = pd.read_csv(results_dir / "results.csv")
    total = 0
    successes = 0
    for xi_deg in data["xi_deg"].unique():
        n = (data["xi_deg"] == xi_deg).sum()
        indices = np.logical_and(data["xi_deg"] == xi_deg, data["failure"].isna())
        t_error = data.loc[indices, "corridor_startpoint_error_mm"].mean()
        rot_error = data.loc[indices, "corridor_angle_error_deg"].mean()
        t_std = data.loc[indices, "corridor_startpoint_error_mm"].std()
        rot_std = data.loc[indices, "corridor_angle_error_deg"].std()
        kwire_tip_error = data.loc[indices, "kwire_tip_error_mm"].mean()
        kwire_rot_error = data.loc[indices, "kwire_angle_error_deg"].mean()
        kwire_tip_std = data.loc[indices, "kwire_tip_error_mm"].std()
        kwire_rot_std = data.loc[indices, "kwire_angle_error_deg"].std()
        success_rate = 100 * indices.sum() / n
        print(
            f"{xi_deg}\\textdegree "
            f"& {n} & {indices.sum()} ({success_rate:.1f}\\%)"
            f"& ${t_error:.1f} \\pm {t_std:.1f}$ & ${rot_error:.1f} \\pm {rot_std:.1f}$ & "
            f"${kwire_tip_error:.1f} \\pm {kwire_tip_std:.1f}$ & ${kwire_rot_error:.1f} \\pm {kwire_rot_std:.1f}$ "
            f"\\\\"
        )

        total += n
        successes += indices.sum()

        # log.info(f"{xi_deg}: {100 * indices.sum() / n:.1f}% success")

    log.info(f"Total: {100 * successes / total:.1f}% success ({successes} / {total})")

    kwargs = dict(linewidth=linewidth, fliersize=1, palette="crest")
    sns.boxplot(
        data,
        x="xi_deg",
        y="corridor_startpoint_error_mm",
        # hue="Rotation Direction",
        **kwargs,
    )
    plt.tight_layout(pad=1.3)
    plt.xlabel(f"Relative Angle ({DEGREES_SIGN})")
    plt.ylabel("Translation Error (mm)")
    plt.ylim(0, 20)
    plt.savefig(images_dir / "triangulate_corridor_startpoint_error.png", dpi=300)
    plt.close()

    sns.boxplot(
        data,
        x="xi_deg",
        y="corridor_angle_error_deg",
        # hue="Rotation Direction",
        **kwargs,
    )
    plt.tight_layout(pad=1.3)
    plt.xlabel(f"Relative Angle ({DEGREES_SIGN})")
    plt.ylabel(f"Rotation Error ({DEGREES_SIGN})")
    plt.ylim(0, 20)
    plt.savefig(images_dir / "triangulate_corridor_angle_error.png", dpi=300)
    plt.close()

    sns.boxplot(
        data,
        x="xi_deg",
        y="kwire_tip_error_mm",
        # hue="Rotation Direction",
        **kwargs,
    )
    plt.tight_layout(pad=1.3)
    plt.xlabel(f"Rotation Angle ({DEGREES_SIGN})")
    plt.ylabel("Translation Error (mm)")
    plt.ylim(0, 5)
    plt.savefig(images_dir / "triangulate_kwire_tip_error.png", dpi=300)
    plt.close()

    sns.boxplot(
        data,
        x="xi_deg",
        y="kwire_angle_error_deg",
        # hue="Rotation Direction",
        **kwargs,
    )
    plt.tight_layout(pad=1.3)
    plt.xlabel(f"Rotation Angle ({DEGREES_SIGN})")
    plt.ylabel(f"Rotation Error ({DEGREES_SIGN})")
    plt.ylim(0, 5)
    plt.savefig(images_dir / "triangulate_kwire_angle_error.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
