import collections
import json
from collections import defaultdict
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from torch import nn

from improve_mesh_segmentation.partnet_grasp.dataset import PartNetGraspDataset, processed_partnet_grasp_generator
from improve_mesh_segmentation.data_correction.correct_sub_partnet import embed
from improve_mesh_segmentation.training.imcnn import SegImcnn


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

og_data_path = "/home/iroberts/projects/VisMeshSegmentation/improve_mesh_segmentation/datasets/partnet_grasp.zip"

model_path = "/home/iroberts/projects/VisMeshSegmentation/run_through/logs/model.zip"




if __name__ == "__main__":
    # Load shared datasets into memory (as lists)
    og_dataset = list(processed_partnet_grasp_generator(og_data_path, set_type=0))
    # corrected_dataset = list(processed_partnet_grasp_generator(corrected_data_path, set_type=0))

    imcnn = SegImcnn(adapt_data=PartNetGraspDataset(og_data_path, set_type=0, only_signal=True))
    imcnn.load_state_dict(torch.load(model_path))
    classification_head = imcnn.model.output_dense
    dropout_prob = 0.5
    stochastic_model = StochasticModel(classification_head, dropout_prob)

    for mesh_idx, ((signal, bc), labels) in enumerate(og_dataset):
        print(".....Correcting Mesh Index: ", mesh_idx)
        og_labels = np.array(labels)
        embeddings = embed(imcnn, [signal, bc])
        # dataset = EmbeddingDataset(embeddings, og_labels)
        # dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)
        #
        # # Get output Distribution
        # predictions = predict_with_uncertainty_batched(stochastic_model, dataloader, n_iter=30)
        # predictions = predictions.numpy()
        #
        # # Calculate entropy
        # a, prob_mat = uncertainty_matrices(predictions)
        # t, e, a = entropy_uncertainty(prob_mat)
        # #
        preds = np.argmax(classification_head(torch.tensor(embeddings)).detach().numpy(),axis=1)
        misclassified_labels_idxs = np.where(preds != og_labels)[0]
        # inc_sorted_unc_idxs = np.argsort(t[misclassified_labels_idxs])
        # inc_sorted_misclassified_labels_idxs = misclassified_labels_idxs[inc_sorted_unc_idxs]

        # Randomly select indices to accept recommendations
        with open('/home/iroberts/projects/VisMeshSegmentation/influence_scores.json', 'r') as f:
            influence_scores = json.load(f)

        influence_dict = {entry['mesh_idx']: entry['influence_per_vertex'] for entry in influence_scores}

        influential_mismatch_idx = []
        for mismatch in misclassified_labels_idxs:
            if abs(influence_dict[mesh_idx][mismatch]) > (np.mean(np.abs(influence_dict[mesh_idx])) + .5*np.std(np.abs(influence_dict[mesh_idx]))):  # 7.565 is global std of influences
                influential_mismatch_idx.append(mismatch)
        sorted_influential_mismatch_idxs = np.array(influential_mismatch_idx)[np.argsort(np.abs(np.array(influence_dict[mesh_idx])[influential_mismatch_idx]))[::-1]]

        correction_percentages = np.arange(10, 110, 10)
        for percent in correction_percentages:
            correction_file_name = "/home/iroberts/projects/VisMeshSegmentation/run_through/corrections/misclassification_influence_baseline_percentage/corrected_labels_misclassificatin_influence_baseline_" + str(percent) + ".csv"

            recommend = percent / 100
            num_recommendations = int(recommend * len(sorted_influential_mismatch_idxs))

            keep_indices = sorted_influential_mismatch_idxs[:num_recommendations]

            if len(keep_indices) > 0:

                with open(correction_file_name, "a") as f:
                    for i, (query_idx) in enumerate(keep_indices):
                        f.write(f"{mesh_idx},{query_idx},{preds[query_idx]}\n")








    # Print summary
    # for method_name, metrics in correction_agreement.items():
    #     print(f"\n=== Correction Agreement: {method_name} ===")
    #     for metric, values in metrics.items():
    #         print(f"{metric.capitalize()}: Mean = {np.mean(values):.3f}, Std = {np.std(values):.3f}")