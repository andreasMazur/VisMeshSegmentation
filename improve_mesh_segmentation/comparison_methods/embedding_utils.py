import numpy as np
import torch

from improve_mesh_segmentation.comparison_methods.helpers import (
    EmbeddingDataset,
    StochasticModel,
    entropy_uncertainty,
    predict_with_uncertainty_batched,
    uncertainty_matrices,
)
from improve_mesh_segmentation.data_correction.correct_sub_partnet import embed


def get_embeddings_labels_preds_with_idx_map(og_dataset, imcnn, classification_head, mesh_range, device="cpu"):
    all_embeddings = []
    all_labels = []
    all_preds = []
    pred_dict = {}
    unc_dict = {}
    mesh_to_global_idx = {}

    global_vertex_idx = 0
    stochastic_model = StochasticModel(classification_head, dropout_prob=0.5).to(device)

    for mesh_idx in mesh_range:
        (signal, bc), labels = og_dataset[mesh_idx]
        labels = np.array(labels)
        embeddings = torch.tensor(embed(imcnn, [signal, bc]))

        with torch.no_grad():
            preds = classification_head(embeddings.to(device))
            preds = np.argmax(preds.detach().cpu().numpy(), axis=1)

        emb_for_unc = embeddings.to(device)
        dataset = EmbeddingDataset(emb_for_unc, labels)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)
        predictions = predict_with_uncertainty_batched(stochastic_model, dataloader, n_iter=30, device=device)
        predictions = predictions.detach().cpu().numpy()

        _, prob_mat = uncertainty_matrices(predictions)
        t, _, _ = entropy_uncertainty(prob_mat)

        num_vertices = embeddings.shape[0]
        current_indices = list(range(global_vertex_idx, global_vertex_idx + num_vertices))
        mesh_to_global_idx[mesh_idx] = current_indices
        global_vertex_idx += num_vertices

        all_embeddings.append(embeddings)
        all_labels.append(labels)
        all_preds.append(preds)
        pred_dict[mesh_idx] = preds
        unc_dict[mesh_idx] = t

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    return all_embeddings, all_labels, all_preds, mesh_to_global_idx, pred_dict, unc_dict


def get_embeddings_labels_preds_with_idx_map_faust(og_dataset, imcnn, classification_head, device="cuda"):
    all_embeddings = []
    all_labels = []
    all_preds = []
    pred_dict = {}
    unc_dict = {}
    mesh_to_global_idx = {}

    global_vertex_idx = 0
    stochastic_model = StochasticModel(classification_head, dropout_prob=0.5).to(device)

    for mesh_idx, ((signal, bc), labels) in enumerate(og_dataset):
        labels = labels.detach().cpu().numpy()
        embeddings = torch.tensor(embed(imcnn, [signal, bc]))

        with torch.no_grad():
            preds = classification_head(embeddings.to(device))
            preds = np.argmax(preds.detach().cpu().numpy(), axis=1)

        dataset = EmbeddingDataset(embeddings.to(device), labels)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)
        predictions = predict_with_uncertainty_batched(stochastic_model, dataloader, n_iter=30, device=device)
        predictions = predictions.detach().cpu().numpy()

        _, prob_mat = uncertainty_matrices(predictions)
        t, _, _ = entropy_uncertainty(prob_mat)

        num_vertices = embeddings.shape[0]
        current_indices = list(range(global_vertex_idx, global_vertex_idx + num_vertices))
        mesh_to_global_idx[mesh_idx] = current_indices
        global_vertex_idx += num_vertices

        all_embeddings.append(embeddings)
        all_labels.append(labels)
        all_preds.append(preds)
        pred_dict[mesh_idx] = preds
        unc_dict[mesh_idx] = t

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    return all_embeddings, all_labels, all_preds, mesh_to_global_idx, pred_dict, unc_dict
