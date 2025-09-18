from improve_mesh_segmentation.comparison_methods.custom_seg_imcnn import CustomSegImcnn
from improve_mesh_segmentation.partnet_grasp.dataset import PartNetGraspDataset

from tqdm import tqdm

import math
import torch
import numpy as np


def remove_candidates(all_candidates, selected_labels_1, selected_labels_2, to_remove_1, to_remove_2):
    """Remove candidates that have been selected in the previous iterations."""
    # Get all selected vertices
    everything_to_remove = torch.cat([selected_labels_1, selected_labels_2, to_remove_1, to_remove_2], dim=0)

    # Remove all selected vertices from candidates
    all_candidates = set([(int(x), int(y)) for x, y in all_candidates])
    everything_to_remove = set([(int(x), int(y)) for x, y in everything_to_remove])

    return torch.tensor(list(all_candidates - everything_to_remove))


def select_candidates(imcnn, dataset, mesh_indices, vertex_candidates, device="cpu"):
    selected_labels = torch.zeros((0, 2), dtype=torch.int64).to(device)
    for mesh_idx, ((signal, bc), gt) in zip(mesh_indices, dataset):
        # Get mesh candidate vertices:
        # mesh_vertex_candidates = {x_i \in Mesh}
        mesh_vertex_candidates = vertex_candidates[vertex_candidates[:, 0] == mesh_idx][:, 1]

        # Predict class of all mesh vertices
        # {y^f(x_i) | x_i \in Mesh}
        pred = imcnn((signal, bc))

        # Filter predictions and ground truth down to candidate vertices:
        # {y^f(x_i) | x_i \in Mesh and x_i \in C_2} and {y^g(x_i) | x_i \in Mesh and x_i \in C_2}
        pred = pred[mesh_vertex_candidates]
        gt = gt[mesh_vertex_candidates]

        # Get vertex indices of correct candidate predictions
        # {i | x_i \in Mesh and x_i \in C_2 and y^f(x_i) == y^g(x_i)}
        correct_predictions = torch.where(pred.argmax(dim=-1) == gt)[0]
        mesh_samples = mesh_vertex_candidates[correct_predictions]

        # Clean label selections
        new_selections = torch.cat(
            [torch.full_like(mesh_samples, mesh_idx).unsqueeze(-1), mesh_samples.unsqueeze(-1)],
            dim=-1
        )
        selected_labels = torch.cat([selected_labels, new_selections], dim=0)
    dataset.reset()
    return selected_labels


def get_vertex_candidates_from_dataset(dataset, available_candidates, mesh_indices):
    """Filters candidate-indices down to what's available in the given dataset."""
    dataset_candidates = torch.zeros((0, 2), dtype=torch.int64)
    for mesh_idx, ((signal, bc), gt) in tqdm(zip(mesh_indices, dataset), desc="Selecting candidates"):
        vertex_indices = torch.arange(signal.shape[0])
        mesh_index = torch.full_like(vertex_indices, mesh_idx)
        mesh_vertex_indices = torch.cat(
            [mesh_index.unsqueeze(dim=-1), vertex_indices.unsqueeze(dim=-1)], dim=-1
        )
        is_available = torch.tensor([(mvi == available_candidates).all(dim=-1).any() for mvi in mesh_vertex_indices])
        dataset_candidates = torch.cat([dataset_candidates, mesh_vertex_indices[is_available]], dim=0)
    dataset.reset()
    return dataset_candidates


def get_all_mesh_vertex_indices(data_path):
    dataset = PartNetGraspDataset(set_type=3, path_to_zip=data_path)
    all_candidates = torch.zeros((0, 2), dtype=torch.int64)
    for mesh_idx, ((signal, bc), gt) in enumerate(dataset):
        mesh_vertex_indices = torch.arange(signal.shape[0])
        mesh_vertex_indices = torch.cat(
            [torch.full_like(mesh_vertex_indices, mesh_idx).unsqueeze(-1), mesh_vertex_indices.unsqueeze(-1)],
            dim=-1
        )
        all_candidates = torch.cat([all_candidates, mesh_vertex_indices], dim=0)
    return all_candidates


def incv(data_path, epochs, remove_ratio=0.1, max_iterations=10):
    """Iterative Noisy Cross-Validation (INCV) method for mesh segmentation.

    Proposed in:
    > Chen, Pengfei, et al. "Understanding and utilizing deep neural networks trained with noisy labels."
    > International conference on machine learning. PMLR, 2019.

    Parameters
    ----------
    data_path: str
        The path to the noisy dataset.
    epochs: int
        The amount of epochs to train a network for.
    remove_ratio: float
        The remove ratio that determines how many samples with high loss values are removed.
    max_iterations: int
        The maximum number of iterations to run the INCV algorithm for.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get all candidates
    all_candidates = get_all_mesh_vertex_indices(data_path)

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
        dataset_1 = PartNetGraspDataset(set_type=3, path_to_zip=data_path, set_indices=mesh_index_set_1, device=device)
        dataset_2 = PartNetGraspDataset(set_type=3, path_to_zip=data_path, set_indices=mesh_index_set_2, device=device)

        # Get vertex candidates from datasets that can be used for loss computation
        vertex_candidates_1 = get_vertex_candidates_from_dataset(dataset_1, all_candidates, mesh_index_set_1).to(device)
        vertex_candidates_2 = get_vertex_candidates_from_dataset(dataset_2, all_candidates, mesh_index_set_2).to(device)

        # Train network on first half of the dataset for E epochs
        imcnn = CustomSegImcnn(adapt_data=PartNetGraspDataset(data_path, set_type=3, only_signal=True, device=device))
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
        imcnn = CustomSegImcnn(adapt_data=PartNetGraspDataset(data_path, set_type=3, only_signal=True, device=device))
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
    all_candidates = set([(int(x), int(y)) for x, y in get_all_mesh_vertex_indices(data_path)])
    selected_candidates = set([(int(x), int(y)) for x, y in all_selected])

    return torch.tensor(list(selected_candidates)), torch.tensor(list(all_candidates - selected_candidates))
