import json
import os

import numpy as np
import scipy as sp
from cleanlab.filter import find_label_issues

from improve_mesh_segmentation.comparison_methods.cv_training import (
    ensure_faust_cv_predictions,
    ensure_partnet_cv_predictions,
)
from improve_mesh_segmentation.comparison_methods.embedding_utils import (
    get_embeddings_labels_preds_with_idx_map,
    get_embeddings_labels_preds_with_idx_map_faust,
)
from improve_mesh_segmentation.comparison_methods.helpers import global_mesh_sort, method_logs_dir, write_label_changes
from improve_mesh_segmentation.comparison_methods.model_utils import (
    get_device,
    load_faust_dataset,
    load_faust_model,
    load_partnet_dataset,
    load_partnet_model,
)


def _build_confident_learning_labels(labels, mesh_map, issues, binary_labels=False):
    result = {}
    new_labels = {}

    for key, values in mesh_map.items():
        issue_positions = [i for i, v in enumerate(values) if v in issues]
        original_labels = labels[values]

        if binary_labels:
            final_labs = np.zeros_like(original_labels, dtype=int)
            final_labs[issue_positions] = 1
            new_labels[key] = final_labs
        else:
            flipped_labels = original_labels.copy()
            flipped_labels[issue_positions] = 1 - flipped_labels[issue_positions]
            new_labels[key] = flipped_labels

        if issue_positions:
            result[key] = issue_positions

    changed_idx_unc = {
        key: [list(issues).index(val) for val in np.array(mesh_map[key])[result[key]]]
        for key in result.keys()
    }
    return result, changed_idx_unc, new_labels


def confident_learning_partnet(
    data_path,
    model_path,
    experiment_directory,
    cv_k=5,
    cv_logging_dir=None,
    cv_epochs=10,
    training_epochs=10,
    mesh_range=range(0, 100),
):
    """Run Confident Learning on PartNet-Grasp.

    The IMCNN and CV out-of-sample predictions are trained automatically when missing.
    Predictions are cached at ``{cv_logging_dir}/{k}/cv_out_of_sample_preds.json``.
    """
    cv_logging_dir = cv_logging_dir or f"{experiment_directory}/cv_logs"
    cv_preds_path = ensure_partnet_cv_predictions(
        data_path=data_path,
        k=cv_k,
        cv_logging_dir=cv_logging_dir,
        n_epochs=cv_epochs,
    )

    device = get_device()
    imcnn, classification_head = load_partnet_model(
        model_path, data_path, device=device, n_epochs=training_epochs,
    )
    og_dataset = load_partnet_dataset(data_path, set_type=3)

    with open(cv_preds_path, "r") as f:
        cv_preds = json.load(f)
    cv_pred_dict = {entry["mesh_idx"]: entry["preds"] for entry in cv_preds}

    embs, labels, preds, mesh_map, _, _ = get_embeddings_labels_preds_with_idx_map(
        og_dataset, imcnn, classification_head, mesh_range, device=str(device),
    )

    missing = [mesh_idx for mesh_idx in mesh_range if mesh_idx not in cv_pred_dict]
    if missing:
        raise RuntimeError(
            f"CV predictions at {cv_preds_path} are missing meshes: {missing[:5]}"
            f"{'...' if len(missing) > 5 else ''}"
        )
    all_preds = [cv_pred_dict[mesh_idx] for mesh_idx in mesh_range]
    cv_all_preds = np.concatenate(all_preds, axis=0)
    pred_probs = sp.special.softmax(cv_all_preds, axis=-1)

    issues = find_label_issues(
        labels=labels,
        pred_probs=pred_probs,
        return_indices_ranked_by="self_confidence",
    )

    result, changed_idx_unc, new_labels = _build_confident_learning_labels(labels, mesh_map, issues, binary_labels=False)
    sorted_by_unc, _ = global_mesh_sort(result, changed_idx_unc)

    output_dir = method_logs_dir(experiment_directory, "confident_learning")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"confident_learning_{cv_k}.csv")
    write_label_changes(output_path, sorted_by_unc, new_labels)
    return output_path


def confident_learning_faust(
    dataset_path,
    segmentation_labels_path,
    logging_dir,
    noise_level,
    model_path,
    experiment_directory,
    dataset_id,
    cv_k=5,
    cv_epochs=10,
    training_epochs=10,
):
    """Run Confident Learning on FAUST.

    The IMCNN and CV out-of-sample predictions are trained automatically when missing.
    Predictions are cached under ``{experiment_directory}/faust_cv_preds/noise_lvl_{noise}/``.
    """
    cv_preds_path = ensure_faust_cv_predictions(
        dataset_path=dataset_path,
        segmentation_labels_path=segmentation_labels_path,
        logging_dir=logging_dir,
        noise_level=noise_level,
        k=cv_k,
        experiment_directory=experiment_directory,
        n_epochs=cv_epochs,
    )

    device = get_device()
    imcnn, classification_head = load_faust_model(
        model_path, dataset_path, segmentation_labels_path, logging_dir, noise_level,
        device=device, n_epochs=training_epochs,
    )
    og_dataset = load_faust_dataset(
        dataset_path, segmentation_labels_path, logging_dir, noise_level, set_type=3, device=device,
    )

    with open(cv_preds_path, "r") as f:
        cv_preds = json.load(f)
    cv_pred_dict = {entry["mesh_idx"]: entry["preds"] for entry in cv_preds}

    embs, labels, preds, mesh_map, _, _ = get_embeddings_labels_preds_with_idx_map_faust(
        og_dataset, imcnn, classification_head, device=str(device),
    )

    mesh_order = list(range(len(og_dataset)))
    missing = [mesh_idx for mesh_idx in mesh_order if mesh_idx not in cv_pred_dict]
    if missing:
        raise RuntimeError(
            f"CV predictions at {cv_preds_path} are missing meshes: {missing[:5]}"
            f"{'...' if len(missing) > 5 else ''}"
        )
    all_preds = [cv_pred_dict[mesh_idx] for mesh_idx in mesh_order]
    cv_all_preds = np.concatenate(all_preds, axis=0)
    pred_probs = sp.special.softmax(cv_all_preds, axis=-1)

    issues = find_label_issues(
        labels=labels,
        pred_probs=pred_probs,
        return_indices_ranked_by="self_confidence",
    )

    result, changed_idx_unc, new_labels = _build_confident_learning_labels(labels, mesh_map, issues, binary_labels=True)
    sorted_by_unc, _ = global_mesh_sort(result, changed_idx_unc)

    output_dir = method_logs_dir(experiment_directory, "confident_learning", dataset_id=dataset_id)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"confident_learning_{cv_k}.csv")
    write_label_changes(output_path, sorted_by_unc, new_labels)
    return output_path
