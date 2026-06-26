import os
import random
from collections import Counter

import numpy as np
import torch

from improve_mesh_segmentation.comparison_methods.cv_training import (
    ensure_faust_cv_models,
    ensure_partnet_cv_models,
)
from improve_mesh_segmentation.comparison_methods.helpers import method_logs_dir, write_label_changes
from improve_mesh_segmentation.comparison_methods.model_utils import (
    FAUST_IMCNN_KWARGS,
    get_device,
    load_faust_dataset,
    load_partnet_dataset,
)
from improve_mesh_segmentation.faust.segmentation_data.faust_segmentation_dataset import FaustSegmentationDataset
from improve_mesh_segmentation.partnet_grasp.dataset import PartNetGraspDataset
from improve_mesh_segmentation.training.imcnn import SegImcnn


def cv_model_pred_partnet(model_path, model_dirs, data_path, signal, bc, labels, voting="majority"):
    num_models = len(model_dirs)
    cv_preds = []

    for model_dir in model_dirs:
        model = f"{model_path}/{model_dir}/model.zip"
        imcnn = SegImcnn(adapt_data=PartNetGraspDataset(data_path, set_type=3, only_signal=True))
        imcnn.load_state_dict(torch.load(model, map_location="cpu"))
        preds = imcnn([signal, bc]).detach().cpu().numpy()
        cv_preds.append(np.argmax(preds, axis=1))

    cv_preds = np.stack(cv_preds, axis=0)
    num_points = cv_preds.shape[1]
    final_preds = np.zeros(num_points, dtype=int)
    misclassified_idxs = []

    for i in range(num_points):
        point_preds = cv_preds[:, i]
        gt_label = labels[i]
        most_common_label, _ = Counter(point_preds).most_common(1)[0]

        if voting == "majority":
            final_preds[i] = most_common_label
            if most_common_label != gt_label:
                misclassified_idxs.append(i)
        else:
            raise ValueError(f"Unknown voting strategy: {voting}")

    return final_preds, misclassified_idxs


def cv_model_pred_faust(model_path, model_dirs, dataset_path, segmentation_labels_path, logging_dir, noise_level,
                        signal, bc, labels, device):
    num_models = len(model_dirs)
    cv_preds = []

    for model_dir in model_dirs:
        model = f"{model_path}/{model_dir}/model.zip"
        imcnn = SegImcnn(
            adapt_data=FaustSegmentationDataset(
                path_to_zip=dataset_path,
                path_to_segmentation_labels=segmentation_labels_path,
                logging_dir=logging_dir,
                set_type=0,
                only_signal=True,
                device=device,
                noise_level=noise_level,
            ),
            **FAUST_IMCNN_KWARGS,
        ).to(device)
        imcnn.load_state_dict(torch.load(model, map_location=device))
        preds = imcnn([signal, bc]).detach().cpu().numpy()
        cv_preds.append(np.argmax(preds, axis=1))

    cv_preds = np.stack(cv_preds, axis=0)
    num_points = cv_preds.shape[1]
    final_preds = np.zeros(num_points, dtype=int)
    misclassified_idxs = []

    for i in range(num_points):
        point_preds = cv_preds[:, i]
        gt_label = labels[i] if not torch.is_tensor(labels) else labels[i].item()
        most_common_label, _ = Counter(point_preds).most_common(1)[0]
        final_preds[i] = most_common_label
        if most_common_label != gt_label:
            misclassified_idxs.append(i)

    return final_preds, misclassified_idxs


def randomly_ranked_mesh_changes(mesh_changes_dict, seed=42):
    if seed is not None:
        random.seed(seed)

    all_changes = []
    for mesh_idx, point_list in mesh_changes_dict.items():
        for point_idx in point_list:
            all_changes.append((mesh_idx, point_idx))

    random.shuffle(all_changes)
    return [(mesh_idx, point_idx, rank) for rank, (mesh_idx, point_idx) in enumerate(all_changes)]


def _list_model_dirs(cv_models_path):
    return sorted(
        d for d in os.listdir(cv_models_path)
        if d.startswith("model_cv_") and os.path.isfile(os.path.join(cv_models_path, d, "model.zip"))
    )


def ensemble_majority_partnet(
    data_path,
    experiment_directory,
    cv_k=5,
    cv_logging_dir=None,
    cv_epochs=10,
    seed=42,
):
    """Run Ensemble Majority baseline on PartNet-Grasp.

    CV models are trained automatically if missing.
    """
    cv_logging_dir = cv_logging_dir or f"{experiment_directory}/cv_logs"
    cv_models_path = ensure_partnet_cv_models(
        data_path=data_path,
        k=cv_k,
        cv_logging_dir=cv_logging_dir,
        n_epochs=cv_epochs,
    )

    dataset = load_partnet_dataset(data_path, set_type=3)
    model_dirs = _list_model_dirs(cv_models_path)
    if not model_dirs:
        raise RuntimeError(f"No CV models found in {cv_models_path}")

    mesh_preds_dict = {}
    mesh_changes_dict = {}
    for mesh_idx, ((signal, bc), labels) in enumerate(dataset):
        labels = np.array(labels)
        mesh_preds, points_to_change = cv_model_pred_partnet(
            cv_models_path, model_dirs, data_path, signal, bc, labels, voting="majority",
        )
        mesh_preds_dict[mesh_idx] = mesh_preds
        mesh_changes_dict[mesh_idx] = points_to_change

    ranked_list = randomly_ranked_mesh_changes(mesh_changes_dict, seed=seed)
    output_dir = method_logs_dir(experiment_directory, "ensemble_majority_baseline")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"ensemble_majority_baseline_{cv_k}.csv")
    write_label_changes(output_path, ranked_list, mesh_preds_dict)
    return output_path


def ensemble_majority_faust(
    dataset_path,
    segmentation_labels_path,
    logging_dir,
    noise_level,
    experiment_directory,
    dataset_id,
    cv_k=5,
    cv_epochs=10,
    seed=42,
):
    """Run Ensemble Majority baseline on FAUST.

    CV models are trained automatically if missing.
    """
    cv_models_path = ensure_faust_cv_models(
        dataset_path=dataset_path,
        segmentation_labels_path=segmentation_labels_path,
        logging_dir=logging_dir,
        noise_level=noise_level,
        k=cv_k,
        n_epochs=cv_epochs,
    )

    device = get_device()
    dataset = load_faust_dataset(
        dataset_path, segmentation_labels_path, logging_dir, noise_level, set_type=3, device=device,
    )
    model_dirs = _list_model_dirs(cv_models_path)
    if not model_dirs:
        raise RuntimeError(f"No CV models found in {cv_models_path}")

    mesh_preds_dict = {}
    mesh_changes_dict = {}
    for mesh_idx, ((signal, bc), labels) in enumerate(dataset):
        labels_np = labels.detach().cpu().numpy() if torch.is_tensor(labels) else np.array(labels)
        mesh_preds, points_to_change = cv_model_pred_faust(
            cv_models_path, model_dirs, dataset_path, segmentation_labels_path, logging_dir, noise_level,
            signal, bc, labels_np, device,
        )
        mesh_preds_dict[mesh_idx] = mesh_preds
        mesh_changes_dict[mesh_idx] = points_to_change

    ranked_list = randomly_ranked_mesh_changes(mesh_changes_dict, seed=seed)
    output_dir = method_logs_dir(experiment_directory, "ensemble_majority_baseline", dataset_id=dataset_id)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"ensemble_majority_baseline_{cv_k}.csv")
    write_label_changes(output_path, ranked_list, mesh_preds_dict)
    return output_path
