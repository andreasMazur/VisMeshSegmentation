import collections
import numpy as np
import torch
from torch import nn




def write_label_changes(path,points_list,preds):
    with open(path, "a") as f:
        for i, (mesh_idx,point_idx,_) in enumerate(points_list):
                f.write(f"{mesh_idx},{point_idx},{preds[mesh_idx][point_idx]}\n")


class StochasticModel(nn.Module):
    def __init__(self, h, dropout_prob=0.5):
        super(StochasticModel, self).__init__()
        # self.g = nn.Sequential(
        #     g,
        #     nn.Dropout(p=dropout_prob)  # Add dropout after g
        # )
        self.h = nn.Sequential(
            nn.Dropout(p=dropout_prob),  # Add dropout before logits
            h
        )

    def forward(self, x):
        # x = self.g(x)
        self.h.train()
        x = self.h(x)
        return x

class EmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


def predict_with_uncertainty_batched(f_model, data_loader, n_iter=10, device="cpu"):
    """
    Perform N stochastic forward passes over batches of data and return predictions.

    Args:
        f_model (nn.Module): The model with dropout layers.
        data_loader (DataLoader): DataLoader for the dataset.
        n_iter (int): Number of stochastic forward passes.
        device (str): Device to perform computations on ('cpu' or 'cuda').

    Returns:
        torch.Tensor: Predictions of shape (n_iter, num_data, num_classes).
    """
    # f_model.train()  # Ensure dropout is active during inference

    # Store predictions for all iterations
    all_preds = []
    for _ in range(n_iter):
        preds = []
        for inputs, _ in data_loader:
            with torch.no_grad():
                batch_preds = torch.softmax(f_model(inputs), dim=1)

            preds.append(batch_preds)
        # Concatenate predictions for this iteration
        all_preds.append(torch.cat(preds, dim=0))

    # Stack predictions across iterations
    return torch.stack(all_preds, dim=0)

def entropy_uncertainty(probs): # three dimentianl array with d1 as datapoints, (d2) the rows as samples and (d3) the columns as probability for each class
    p = np.array(probs)
    entropy = -p*np.ma.log10(p)
    entropy = entropy.filled(0)
    a = np.sum(entropy, axis=1)
    a = np.sum(a, axis=1) / entropy.shape[1]
    p_m = np.mean(p, axis=1)
    total = -np.sum(p_m*np.ma.log10(p_m), axis=1)
    total = total.filled(0)
    e = total - a
    return total, e, a

def uncertainty_matrices(predictions):
    pred = predictions

    predictions_temp = [[[] for j in range(predictions.shape[0])] for i in range(predictions.shape[1])]
    prob_matrix = a = [[[] for j in range(predictions.shape[0])] for i in range(predictions.shape[1])]


    for model_index, model_prediction in enumerate(pred):
        for data_index in range(predictions.shape[1]):
            prob_matrix[data_index][model_index] = model_prediction[data_index]
            predictions_temp[data_index][model_index] = np.argmax(model_prediction[data_index])

    prediction_list = []
    for prob_predic_data in predictions_temp:
        counter = collections.Counter(prob_predic_data)
        temp = collections.Counter(counter)
        prediction_list.append(temp.most_common()[0][0])

    return np.array(prediction_list), np.array(prob_matrix)