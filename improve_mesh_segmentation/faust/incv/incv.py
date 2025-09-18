from improve_mesh_segmentation.comparison_methods.custom_seg_imcnn import CustomSegImcnn
from improve_mesh_segmentation.comparison_methods.incv import (
    get_vertex_candidates_from_dataset,
    select_candidates,
    remove_candidates
)
from improve_mesh_segmentation.faust.segmentation_data.faust_segmentation_dataset import FaustSegmentationDataset

import torch
import os
import math
import numpy as np


def get_all_mesh_vertex_indices(dataset_path, segmentation_labels_path, logging_dir):
    dataset = FaustSegmentationDataset(
        path_to_zip=dataset_path,
        path_to_segmentation_labels=segmentation_labels_path,
        logging_dir=logging_dir,
        set_type=3,
        only_signal=False,
        device=None,
        noise_level=None
    )

    all_candidates = torch.zeros((0, 2), dtype=torch.int64)
    for mesh_idx, ((signal, bc), gt) in enumerate(dataset):
        mesh_vertex_indices = torch.arange(signal.shape[0])
        mesh_vertex_indices = torch.cat(
            [torch.full_like(mesh_vertex_indices, mesh_idx).unsqueeze(-1), mesh_vertex_indices.unsqueeze(-1)],
            dim=-1
        )
        all_candidates = torch.cat([all_candidates, mesh_vertex_indices], dim=0)
    return all_candidates


def incv(data_path, segmentation_labels_path, logging_dir, epochs, remove_ratio=0.1, max_iterations=10):
    """Iterative Noisy Cross-Validation (INCV) method for mesh segmentation.

    Proposed in:
    > Chen, Pengfei, et al. "Understanding and utilizing deep neural networks trained with noisy labels."
    > International conference on machine learning. PMLR, 2019.

    Parameters
    ----------
    data_path: str
        The path to the noisy dataset.
    segmentation_labels_path: str
        The path to the clean segmentation labels.
    logging_dir: str
        The path to the logging directory in which the noisy segmentation labels are stored.
    epochs: int
        The amount of epochs to train a network for.
    remove_ratio: float
        The remove ratio that determines how many samples with high loss values are removed.
    max_iterations: int
        The maximum number of iterations to run the INCV algorithm for.
    """
    assert os.path.isfile(f"{logging_dir}/noisy_segmentation_labels.npy"), "Noisy segmentation labels not found!"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get all candidates
    all_candidates = get_all_mesh_vertex_indices(
        dataset_path=data_path,
        segmentation_labels_path=segmentation_labels_path,
        logging_dir=logging_dir
    )

    # Get set of clean labels
    all_selected = torch.zeros((0, 2), dtype=torch.int64)
    for iterations in range(max_iterations):
        print(
            f"\nCurrently in iteration: {iterations + 1}/{max_iterations} | "
            f"Available candidates: {all_candidates.shape[0]} | "
            f"Made selections: {all_selected.shape[0]}"
        )
        # Randomly divide dataset into 2 halves (over meshes - not vertices)
        shuffled_mesh_indices = np.arange(100)
        np.random.shuffle(shuffled_mesh_indices)
        mesh_index_set_1 = shuffled_mesh_indices[:50]
        mesh_index_set_2 = shuffled_mesh_indices[50:]
        dataset_1 = FaustSegmentationDataset(
            path_to_zip=data_path,
            path_to_segmentation_labels=segmentation_labels_path,
            logging_dir=logging_dir,
            set_type=3,
            only_signal=False,
            device=device,
            noise_level=None,
            set_indices=mesh_index_set_1
        )
        dataset_2 = FaustSegmentationDataset(
            path_to_zip=data_path,
            path_to_segmentation_labels=segmentation_labels_path,
            logging_dir=logging_dir,
            set_type=3,
            only_signal=False,
            device=device,
            noise_level=None,
            set_indices=mesh_index_set_2
        )

        # Get vertex candidates from datasets that can be used for loss computation
        vertex_candidates_1 = get_vertex_candidates_from_dataset(dataset_1, all_candidates, mesh_index_set_1).to(device)
        vertex_candidates_2 = get_vertex_candidates_from_dataset(dataset_2, all_candidates, mesh_index_set_2).to(device)

        # Train network on first half of the dataset for E epochs
        imcnn = CustomSegImcnn(
            adapt_data=FaustSegmentationDataset(
                path_to_zip=data_path,
                path_to_segmentation_labels=segmentation_labels_path,
                logging_dir=logging_dir,
                set_type=3,
                only_signal=True,
                device=device,
                noise_level=None
            ),
            signal_dim=544,
            kernel_size=(3, 6),
            segmentation_classes=8,
            template_radius=0.027744965069279016,
            layer_conf=[(32, 6), (32, 6)]
        )
        imcnn.to(device)
        for epoch in range(epochs):
            train_history_1 = imcnn.train_loop(
                dataset=dataset_1,
                mesh_indices=torch.tensor(mesh_index_set_1).to(device),
                candidate_indices=torch.cat([all_selected.to(device), vertex_candidates_1], dim=0),
                optimizer=torch.optim.Adam(imcnn.parameters()),
                verbose=True,
                epoch=epoch
            )
            dataset_1.reset()

        # Predict labels in second half of the dataset
        selected_labels_1 = select_candidates(
            imcnn,
            dataset_2,
            torch.tensor(mesh_index_set_2).to(device),
            vertex_candidates_2,
            device=device
        )

        # Determine n = r * |vertex_candidates_1| samples to remove
        n = math.floor(remove_ratio * selected_labels_1.shape[0])
        r_1 = torch.cat(train_history_1["candidate_loss_values"], dim=0)
        r_1 = r_1[r_1[:, -1].argsort(descending=True)][:n, :-1]

        # Train network on second half of the dataset for E epochs
        imcnn = CustomSegImcnn(
            adapt_data=FaustSegmentationDataset(
                path_to_zip=data_path,
                path_to_segmentation_labels=segmentation_labels_path,
                logging_dir=logging_dir,
                set_type=3,
                only_signal=True,
                device=device,
                noise_level=None
            ),
            signal_dim=544,
            kernel_size=(3, 6),
            segmentation_classes=8,
            template_radius=0.027744965069279016,
            layer_conf=[(32, 6), (32, 6)]
        )
        imcnn.to(device)
        for epoch in range(epochs):
            train_history_2 = imcnn.train_loop(
                dataset=dataset_2,
                mesh_indices=torch.tensor(mesh_index_set_2).to(device),
                candidate_indices=torch.cat([all_selected.to(device), vertex_candidates_2], dim=0),
                optimizer=torch.optim.Adam(imcnn.parameters()),
                verbose=True,
                epoch=epoch
            )
            dataset_2.reset()

        # Predict labels in first half of the dataset
        selected_labels_2 = select_candidates(
            imcnn,
            dataset_1,
            torch.tensor(mesh_index_set_1).to(device),
            vertex_candidates_1,
            device=device
        )

        # Determine n = r * |vertex_candidates_1| samples to remove
        n = math.floor(remove_ratio * selected_labels_2.shape[0])
        r_2 = torch.cat(train_history_2["candidate_loss_values"], dim=0)
        r_2 = r_2[r_2[:, -1].argsort(descending=True)][:n, :-1]

        all_selected = torch.cat(
            [all_selected.to(device), selected_labels_1.to(device), selected_labels_2.to(device)], dim=0
        )
        all_candidates = remove_candidates(all_candidates, selected_labels_1, selected_labels_2, r_1, r_2)
        if all_candidates.shape[0] == 0:
            break

    # Get all candidates
    print(
        f"\nFINAL | "
        f"Available candidates: {all_candidates.shape[0]} | "
        f"Made selections: {all_selected.shape[0]}"
    )
    all_candidates = set(
        [(int(x), int(y)) for x, y in get_all_mesh_vertex_indices(data_path, segmentation_labels_path, logging_dir)]
    )
    selected_candidates = set([(int(x), int(y)) for x, y in all_selected])

    return torch.tensor(list(selected_candidates)), torch.tensor(list(all_candidates - selected_candidates))
