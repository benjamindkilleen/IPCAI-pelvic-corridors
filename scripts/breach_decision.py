import numpy as np
from pathlib import Path
from cortical_breach_detection import utils
from cortical_breach_detection.utils import geo_utils
from deepdrr import geo
import logging
from rich.logging import RichHandler
from rich.progress import track
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

log = logging.getLogger(__name__)
logging.basicConfig(handlers=[RichHandler()])
log.setLevel(logging.DEBUG)

np.random.seed(0)


def get_data_line(
    front_disk: geo.Point3D,
    guide_vector: geo.Vector3D,
    startpoint: geo.Point3D,
    w_in_world: geo.Vector3D,
    index_from_world: geo.Transform,
):
    front_disk_index = index_from_world @ front_disk
    startpoint_index = index_from_world @ startpoint
    guide_vector_index = geo.vector(
        index_from_world @ (front_disk + guide_vector) - front_disk_index
    ).hat()
    w_in_index = geo.vector(index_from_world @ (startpoint + w_in_world) - startpoint_index).hat()

    startpoint_to_kwire_distance_index = geo_utils.geo_distance_to_line(
        startpoint_index,
        front_disk_index,
        guide_vector_index,
    )
    relative_angle_index = guide_vector_index.angle(w_in_index)

    startpoint_to_kwire_distance = geo_utils.geo_distance_to_line(
        startpoint, front_disk, guide_vector
    )
    midpoint_to_kwire_distance = geo_utils.geo_distance_to_line(
        startpoint + 50 * w_in_world, front_disk, guide_vector
    )
    endpoint_to_kwire_distance = geo_utils.geo_distance_to_line(
        startpoint + 120 * w_in_world, front_disk, guide_vector
    )
    relative_angle = guide_vector.angle(w_in_world)
    return [
        startpoint_to_kwire_distance_index,
        relative_angle_index,
        startpoint_to_kwire_distance,
        midpoint_to_kwire_distance,
        endpoint_to_kwire_distance,
        relative_angle,
    ]


SIZE = 768


def main():
    run_dir = Path(
        # No fine-tuning
        # "/home/killeen/projects/cortical-breach-detection/results/test/2022-05-90_09-33-09_yoda_full-but-bad"
        # After some fine-tuning
        "/home/killeen/projects/cortical-breach-detection/results/test/2022-05-10/11-56-38"
    )
    predictions_dir = list(run_dir.glob("test_*_predictions"))[-1]
    gt_data = []
    data = []
    labels = []
    oop = []
    for pred_info_path in track(list(predictions_dir.glob("*.json"))):
        pred_info = utils.load_json(pred_info_path)
        cortical_breach_label = pred_info["cortical_breach_label"]
        base = pred_info["base"]
        front_disk = geo.point(pred_info["front_disk"])
        guide_vector = geo.vector(pred_info["guide_vector"])
        pred_startpoint = geo.point(pred_info["pred_startpoint"])
        w_in_world = geo.vector(pred_info["w_in_world"])

        gt_front_disk = geo.point(pred_info["gt_front_disk"])
        gt_guide_vector = geo.vector(pred_info["gt_guide_vector"])
        gt_startpoint = geo.point(pred_info["gt_startpoint"])
        gt_w_in_world = geo.vector(pred_info["gt_w_in_world"])

        index_from_world = geo.Transform(np.array(pred_info["index_from_world"]))

        if "gt_endpoint" in pred_info:
            gt_endpoint = geo.point(pred_info["gt_endpoint"])
            gt_endpoint_in_index = index_from_world @ gt_endpoint
            if (
                gt_endpoint_in_index[1] < 0
                or gt_endpoint_in_index[1] >= SIZE
                or gt_endpoint_in_index[0] < 0
                or gt_endpoint_in_index[0] > SIZE
            ):
                continue

        data.append(
            get_data_line(front_disk, guide_vector, pred_startpoint, w_in_world, index_from_world)
        )
        gt_data.append(
            get_data_line(
                gt_front_disk, gt_guide_vector, gt_startpoint, gt_w_in_world, index_from_world
            )
        )
        labels.append(cortical_breach_label)

    indexing = np.logical_not(np.isnan(data).any(axis=1))
    data = np.array(data)[indexing]
    gt_data = np.array(gt_data)[indexing]
    labels = np.array(labels)[indexing]

    split = len(data) // 3
    train_idx = np.random.choice(len(data), size=split, replace=False)
    test_idx = np.setdiff1d(np.arange(len(data)), train_idx)
    estimator = RandomForestClassifier().fit(data[train_idx], labels[train_idx])
    preds = estimator.predict(data)
    log.info(f"classification_report:\n{classification_report(labels[test_idx], preds[test_idx])}")
    log.info(f"confusion_matrix:\n{confusion_matrix(labels[test_idx], preds[test_idx])}")

    # estimator = RandomForestClassifier().fit(gt_data, labels)
    # preds = estimator.predict(data)
    # log.info(f"classification_report:\n{classification_report(labels, preds)}")
    # log.info(f"confusion_matrix:\n{confusion_matrix(labels, preds)}")

    log.info("for oop perturbation")
    indexing = np.logical_and(
        np.isin(np.arange(data.shape[0]), test_idx), gt_data[:, 1] < np.radians(5)
    )
    oop_data = data[indexing]
    oop_labels = labels[indexing]
    oop_preds = estimator.predict(oop_data)
    log.info(f"classification_report:\n{classification_report(oop_labels, oop_preds)}")
    log.info(f"confusion_matrix:\n{confusion_matrix(oop_labels, oop_preds)}")


if __name__ == "__main__":
    main()
