"""Cross-validation model training and prediction generation.

Run these utilities before Confident Learning (M_2) or Ensemble Majority (M_4).
"""

import json
import os

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch import nn

from improve_mesh_segmentation.comparison_methods.model_utils import FAUST_IMCNN_KWARGS, get_device
from improve_mesh_segmentation.faust.segmentation_data.faust_segmentation_dataset import FaustSegmentationDataset
from improve_mesh_segmentation.partnet_grasp.dataset import PartNetGraspDataset, processed_partnet_grasp_generator
from improve_mesh_segmentation.training.imcnn import SegImcnn
from improve_mesh_segmentation.training.train_logging import log_training

DATASET_LENGTH = 100


def partnet_cv_models_dir(cv_logging_dir, k):
    return os.path.join(cv_logging_dir, str(k))


def faust_cv_models_dir(logging_dir, k):
    return os.path.join(logging_dir, "cv_models", str(k))


def partnet_cv_preds_path(cv_logging_dir, k):
    return os.path.join(partnet_cv_models_dir(cv_logging_dir, k), "cv_out_of_sample_preds.json")


def faust_cv_preds_path(experiment_directory, noise_level, k):
    return os.path.join(
        experiment_directory,
        "faust_cv_preds",
        f"noise_lvl_{noise_level:.3f}",
        str(k),
        "cv_out_of_sample_preds.json",
    )


def _has_trained_models(cv_models_path, k):
    if not os.path.isdir(cv_models_path):
        return False
    model_dirs = sorted(d for d in os.listdir(cv_models_path) if d.startswith("model_cv_"))
    if len(model_dirs) != k:
        return False
    return all(
        os.path.isfile(os.path.join(cv_models_path, model_dir, "model.zip"))
        for model_dir in model_dirs
    )


def ensure_partnet_cv_models(data_path, k, cv_logging_dir, n_epochs=10, device=None, verbose=False):
    """Return path to CV models, training them first if they are missing."""
    cv_models_path = partnet_cv_models_dir(cv_logging_dir, k)
    if _has_trained_models(cv_models_path, k):
        return cv_models_path

    print(f"CV models not found at {cv_models_path}. Training {k}-fold CV models...")
    os.makedirs(cv_logging_dir, exist_ok=True)
    train_partnet_cv_models(
        data_path=data_path,
        k=k,
        n_epochs=n_epochs,
        logging_dir=cv_logging_dir,
        device=device,
        verbose=verbose,
    )
    if not _has_trained_models(cv_models_path, k):
        raise RuntimeError(f"CV model training finished but models are still missing at {cv_models_path}")
    return cv_models_path


def ensure_partnet_cv_predictions(
    data_path, k, cv_logging_dir, n_epochs=10, device=None, verbose=False,
):
    """Return path to out-of-sample CV predictions, creating models and preds if needed."""
    cv_models_path = ensure_partnet_cv_models(
        data_path, k, cv_logging_dir, n_epochs=n_epochs, device=device, verbose=verbose,
    )
    preds_path = partnet_cv_preds_path(cv_logging_dir, k)
    if os.path.isfile(preds_path):
        return preds_path

    print(f"CV predictions not found at {preds_path}. Generating out-of-sample predictions...")
    generate_partnet_cv_predictions(
        data_path=data_path,
        cv_models_path=cv_models_path,
        output_path=preds_path,
        k=k,
        device=device,
    )
    return preds_path


def ensure_faust_cv_models(
    dataset_path,
    segmentation_labels_path,
    logging_dir,
    noise_level,
    k,
    n_epochs=10,
    device=None,
    verbose=False,
):
    """Return path to FAUST CV models, training them first if they are missing."""
    cv_models_path = faust_cv_models_dir(logging_dir, k)
    if _has_trained_models(cv_models_path, k):
        return cv_models_path

    print(f"FAUST CV models not found at {cv_models_path}. Training {k}-fold CV models...")
    train_faust_cv_models(
        dataset_path=dataset_path,
        segmentation_labels_path=segmentation_labels_path,
        logging_dir=logging_dir,
        noise_level=noise_level,
        k=k,
        n_epochs=n_epochs,
        device=device,
        verbose=verbose,
    )
    if not _has_trained_models(cv_models_path, k):
        raise RuntimeError(f"FAUST CV model training finished but models are still missing at {cv_models_path}")
    return cv_models_path


def ensure_faust_cv_predictions(
    dataset_path,
    segmentation_labels_path,
    logging_dir,
    noise_level,
    k,
    experiment_directory,
    n_epochs=10,
    device=None,
    verbose=False,
):
    """Return path to FAUST out-of-sample CV predictions, creating models and preds if needed."""
    cv_models_path = ensure_faust_cv_models(
        dataset_path=dataset_path,
        segmentation_labels_path=segmentation_labels_path,
        logging_dir=logging_dir,
        noise_level=noise_level,
        k=k,
        n_epochs=n_epochs,
        device=device,
        verbose=verbose,
    )
    preds_path = faust_cv_preds_path(experiment_directory, noise_level, k)
    if os.path.isfile(preds_path):
        return preds_path

    print(f"FAUST CV predictions not found at {preds_path}. Generating out-of-sample predictions...")
    generate_faust_cv_predictions(
        dataset_path=dataset_path,
        segmentation_labels_path=segmentation_labels_path,
        logging_dir=logging_dir,
        noise_level=noise_level,
        cv_models_path=cv_models_path,
        output_path=preds_path,
        k=k,
        device=device,
    )
    return preds_path


def train_partnet_cv_models(data_path, k, n_epochs, logging_dir, device=None, skip_validation=True, verbose=False):
    """Train k-fold CV models for PartNet-Grasp and store them under ``logging_dir/{k}/``."""
    device = device or get_device()
    os.makedirs(logging_dir, exist_ok=True)
    fold_logging_dir = partnet_cv_models_dir(logging_dir, k)
    os.makedirs(fold_logging_dir, exist_ok=True)

    idx_folds = np.split(np.arange(DATASET_LENGTH), indices_or_sections=k)
    for fold_idx, test_indices in enumerate(idx_folds):
        train_indices = list(np.array([idx_folds[x] for x in range(k) if x != fold_idx]).flatten())
        model_dir = os.path.join(fold_logging_dir, f"model_cv_{fold_idx}")
        os.makedirs(model_dir, exist_ok=True)

        adapt_data = PartNetGraspDataset(data_path, set_type=0, only_signal=True, set_indices=train_indices)
        train_data = PartNetGraspDataset(data_path, set_type=0, set_indices=train_indices)
        test_data = PartNetGraspDataset(data_path, set_type=2, set_indices=test_indices)

        model = SegImcnn(adapt_data=adapt_data).to(device)
        train_hist = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "test_loss": [],
            "test_accuracy": [],
        }

        for epoch in range(n_epochs):
            train_data.reset()
            epoch_train_hist = model.train_loop(
                dataset=train_data,
                loss_fn=nn.CrossEntropyLoss(),
                optimizer=torch.optim.Adam(model.parameters()),
                verbose=verbose,
                epoch=epoch,
            )
            train_hist["train_loss"].append(float(epoch_train_hist["epoch_loss"].detach()))
            train_hist["train_accuracy"].append(float(epoch_train_hist["epoch_accuracy"].detach()))

        epoch_test_hist = model.validation_loop(dataset=test_data, loss_fn=nn.CrossEntropyLoss(), verbose=False)
        train_hist["test_loss"].append(float(epoch_test_hist["val_epoch_loss"].detach()))
        train_hist["test_accuracy"].append(float(epoch_test_hist["val_epoch_accuracy"].detach()))
        log_training(model, train_hist, model_dir, skip_validation=True, skip_testing=False, verbose=verbose)

    return fold_logging_dir


def generate_partnet_cv_predictions(data_path, cv_models_path, output_path, k=5, device=None):
    """Generate out-of-sample CV predictions for Confident Learning on PartNet-Grasp."""
    device = device or get_device()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    model_dirs = sorted([d for d in os.listdir(cv_models_path) if d.startswith("model_cv_")])
    cv = KFold(n_splits=k)
    cv_pred_dict = []

    for (train_idxs, test_idxs), model_dir in zip(cv.split(np.arange(DATASET_LENGTH)), model_dirs):
        model = os.path.join(cv_models_path, model_dir, "model.zip")
        imcnn = SegImcnn(adapt_data=PartNetGraspDataset(data_path, set_type=3, only_signal=True))
        imcnn.load_state_dict(torch.load(model, map_location=device))
        test_dataset = list(processed_partnet_grasp_generator(data_path, set_type=3, set_indices=test_idxs))

        for mesh_idx, ((signal, bc), _) in zip(test_idxs, test_dataset):
            preds = imcnn([signal, bc]).detach().cpu().numpy().tolist()
            cv_pred_dict.append({"mesh_idx": int(mesh_idx), "preds": preds})

    with open(output_path, "w") as f:
        json.dump(cv_pred_dict, f)
    return output_path


def train_faust_cv_models(
    dataset_path,
    segmentation_labels_path,
    logging_dir,
    noise_level,
    k,
    n_epochs,
    device=None,
    verbose=False,
):
    """Train k-fold CV models for FAUST and store them under ``logging_dir/cv_models/{k}/``."""
    device = device or get_device()
    fold_logging_dir = faust_cv_models_dir(logging_dir, k)
    os.makedirs(fold_logging_dir, exist_ok=True)

    idx_folds = np.split(np.arange(DATASET_LENGTH), indices_or_sections=k)
    for fold_idx, test_indices in enumerate(idx_folds):
        train_indices = list(np.array([idx_folds[x] for x in range(k) if x != fold_idx]).flatten())
        model_dir = os.path.join(fold_logging_dir, f"model_cv_{fold_idx}")
        os.makedirs(model_dir, exist_ok=True)

        adapt_data = FaustSegmentationDataset(
            path_to_zip=dataset_path,
            path_to_segmentation_labels=segmentation_labels_path,
            logging_dir=logging_dir,
            set_type=0,
            only_signal=True,
            device=device,
            noise_level=noise_level,
            set_indices=train_indices,
        )
        train_data = FaustSegmentationDataset(
            path_to_zip=dataset_path,
            path_to_segmentation_labels=segmentation_labels_path,
            logging_dir=logging_dir,
            set_type=0,
            only_signal=False,
            device=device,
            noise_level=noise_level,
            set_indices=train_indices,
        )
        test_data = FaustSegmentationDataset(
            path_to_zip=dataset_path,
            path_to_segmentation_labels=segmentation_labels_path,
            logging_dir=logging_dir,
            set_type=2,
            only_signal=False,
            device=device,
            noise_level=noise_level,
            set_indices=test_indices,
        )

        model = SegImcnn(adapt_data=adapt_data, **FAUST_IMCNN_KWARGS).to(device)
        train_hist = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "test_loss": [],
            "test_accuracy": [],
        }

        for epoch in range(n_epochs):
            train_data.reset()
            epoch_train_hist = model.train_loop(
                dataset=train_data,
                loss_fn=nn.CrossEntropyLoss(),
                optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
                verbose=verbose,
                epoch=epoch,
            )
            train_hist["train_loss"].append(float(epoch_train_hist["epoch_loss"].detach()))
            train_hist["train_accuracy"].append(float(epoch_train_hist["epoch_accuracy"].detach()))

        epoch_test_hist = model.validation_loop(dataset=test_data, loss_fn=nn.CrossEntropyLoss(), verbose=False)
        train_hist["test_loss"].append(float(epoch_test_hist["val_epoch_loss"].detach()))
        train_hist["test_accuracy"].append(float(epoch_test_hist["val_epoch_accuracy"].detach()))
        log_training(model, train_hist, model_dir, skip_validation=True, skip_testing=False, verbose=verbose)

    return fold_logging_dir


def generate_faust_cv_predictions(
    dataset_path,
    segmentation_labels_path,
    logging_dir,
    noise_level,
    cv_models_path,
    output_path,
    k=5,
    device=None,
):
    """Generate out-of-sample CV predictions for Confident Learning on FAUST."""
    device = device or get_device()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    model_dirs = sorted([d for d in os.listdir(cv_models_path) if d.startswith("model_cv_")])
    cv = KFold(n_splits=k)
    cv_pred_dict = []

    for (train_idxs, test_idxs), model_dir in zip(cv.split(np.arange(DATASET_LENGTH)), model_dirs):
        model = os.path.join(cv_models_path, model_dir, "model.zip")
        faust_data = FaustSegmentationDataset(
            path_to_zip=dataset_path,
            path_to_segmentation_labels=segmentation_labels_path,
            logging_dir=logging_dir,
            set_type=3,
            only_signal=False,
            device=device,
            noise_level=noise_level,
            set_indices=test_idxs,
        )
        test_dataset = list(faust_data.dataset)

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

        for mesh_idx, ((signal, bc), _) in zip(test_idxs, test_dataset):
            preds = imcnn([signal, bc]).detach().cpu().numpy().tolist()
            cv_pred_dict.append({"mesh_idx": int(mesh_idx), "preds": preds})

    with open(output_path, "w") as f:
        json.dump(cv_pred_dict, f)
    return output_path
