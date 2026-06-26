import collections
import os

import numpy as np
import torch
from torch import nn


def write_label_changes(path, points_list, preds):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for mesh_idx, point_idx, _ in points_list:
            f.write(f"{mesh_idx},{point_idx},{preds[mesh_idx][point_idx]}\n")


def method_logs_dir(experiment_directory, method_name, dataset_id=None):
    if dataset_id is None:
        return f"{experiment_directory}/{method_name}_logs"
    return f"{experiment_directory}/{dataset_id}/{method_name}_logs"


class StochasticModel(nn.Module):
    def __init__(self, h, dropout_prob=0.5):
        super().__init__()
        self.h = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            h
        )

    def forward(self, x):
        self.h.train()
        return self.h(x)


class EmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


def embedding_entropy_uncertainty(data, labels, model, n_iter=30):
    """MC-dropout entropy per vertex for local correction methods."""
    device = next(model.parameters()).device
    if torch.is_tensor(data):
        embeddings = data.to(device)
    else:
        embeddings = torch.tensor(data, dtype=torch.float32, device=device)

    dataset = EmbeddingDataset(embeddings, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)
    predictions = predict_with_uncertainty_batched(model, dataloader, n_iter=n_iter, device=device)
    predictions = predictions.detach().cpu().numpy()

    _, prob_mat = uncertainty_matrices(predictions)
    total, _, _ = entropy_uncertainty(prob_mat)
    return total


def predict_with_uncertainty_batched(f_model, data_loader, n_iter=10, device="cpu"):
    all_preds = []
    for _ in range(n_iter):
        preds = []
        for inputs, _ in data_loader:
            with torch.no_grad():
                batch_preds = torch.softmax(f_model(inputs), dim=1)
            preds.append(batch_preds)
        all_preds.append(torch.cat(preds, dim=0))
    return torch.stack(all_preds, dim=0)


def entropy_uncertainty(probs):
    p = np.array(probs)
    entropy = -p * np.ma.log10(p)
    entropy = entropy.filled(0)
    aleatoric = np.sum(entropy, axis=1)
    aleatoric = np.sum(aleatoric, axis=1) / entropy.shape[1]
    p_m = np.mean(p, axis=1)
    total = -np.sum(p_m * np.ma.log10(p_m), axis=1)
    total = total.filled(0)
    epistemic = total - aleatoric
    return total, epistemic, aleatoric


def global_mesh_sort(neighbors_dict, values_dict, descending=False):
    combined = []
    for mesh_idx in neighbors_dict:
        vertex_indices = neighbors_dict[mesh_idx]
        values = values_dict[mesh_idx]
        for v_idx, val in zip(vertex_indices, values):
            combined.append((mesh_idx, v_idx, val))
    combined_sorted = sorted(combined, key=lambda x: x[2], reverse=descending)
    return combined_sorted, combined


def uncertainty_matrices(predictions):
    predictions_temp = [[[] for _ in range(predictions.shape[0])] for _ in range(predictions.shape[1])]
    prob_matrix = [[[] for _ in range(predictions.shape[0])] for _ in range(predictions.shape[1])]

    for model_index, model_prediction in enumerate(predictions):
        for data_index in range(predictions.shape[1]):
            prob_matrix[data_index][model_index] = model_prediction[data_index]
            predictions_temp[data_index][model_index] = np.argmax(model_prediction[data_index])

    prediction_list = []
    for prob_predic_data in predictions_temp:
        counter = collections.Counter(prob_predic_data)
        prediction_list.append(counter.most_common()[0][0])

    return np.array(prediction_list), np.array(prob_matrix)
