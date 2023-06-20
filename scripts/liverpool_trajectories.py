from deepdrr import geo
from rich.logging import RichHandler

import logging
import numpy as np
import os

logging.basicConfig(
    level="NOTSET",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

log = logging.getLogger("cortical")

DEGREE_SIGN = "\N{DEGREE SIGN}"


def main():
    wire_left_a = geo.p(143.5100860595703, 11.711020469665527, 53.3400344848632)
    wire_left_b = geo.p(227.31468200683597, 5.674304008483887, 123.8902359008789)
    wire_right_a = geo.p(-41.82706832885742, 9.231474876403809, 31.882556915283203)
    wire_right_b = geo.p(-190.4175720214844, -21.08473777770996, 119.90953063964844)
    gt_left_a = geo.p(148.24375915527344, 12.644296646118164, 57.61609649658203)
    gt_left_b = geo.p(56.51036071777344, 21.813840866088867, -18.84722328186035)
    gt_right_a = geo.p(48.25075149536133, 27.317087173461918, -18.42646598815918)
    gt_right_b = geo.p(-53.25304794311524, 6.946651458740234, 42.43706130981445)

    wire_left = geo.line(wire_left_a, wire_left_b)
    wire_right = geo.line(wire_right_a, wire_right_b)
    gt_left = geo.line(gt_left_a, gt_left_b)
    gt_right = geo.line(gt_right_a, gt_right_b)

    left_angle_error = np.degrees(wire_left.angle(gt_left))
    left_translation_error = min(gt_left.distance(wire_left_a), gt_left.distance(wire_left_b))

    right_angle_error = np.degrees(wire_right.angle(gt_right))
    right_translation_error = min(gt_right.distance(wire_right_a), gt_right.distance(wire_right_b))

    log.info(f"Left error: {left_translation_error:.2f} mm, {left_angle_error:.2f}{DEGREE_SIGN}")
    log.info(f"Right error: {right_translation_error:.2f} mm, {right_angle_error:.2f}{DEGREE_SIGN}")


if __name__ == "__main__":
    main()
