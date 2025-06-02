import json

import numpy as np
from sklearn.cluster import KMeans

from helper_functions import *
from relabel_helpers.deepview_label_corrections import DeepViewLabelRevisit

def influence_baseline(data,mesh_idx):
    idxs = np.array(list(range(len(data))))
    # Randomly select indices to accept recommendations
    with open('/home/iroberts/projects/VisMeshSegmentation/influence_scores1.json', 'r') as f:
        influence_scores = json.load(f)

    influence_dict = {entry['mesh_idx']: entry['influence_per_vertex'] for entry in influence_scores}


    influential_values = np.abs(np.array(influence_dict[mesh_idx]))


    return idxs, influential_values

def influence_uncertainty_combination_baseline(data,mesh_idx,model,labels):
    idxs = np.array(list(range(len(data))))
    # Randomly select indices to accept recommendations
    with open('/home/iroberts/projects/VisMeshSegmentation/influence_scores1.json', 'r') as f:
        influence_scores = json.load(f)

    influence_dict = {entry['mesh_idx']: entry['influence_per_vertex'] for entry in influence_scores}


    influential_values = np.abs(np.array(influence_dict[mesh_idx]))
    # inf_min, inf_max = np.min(influential_values), np.max(influential_values)
    #
    # inf_norm = (influential_values - inf_min) / (inf_max - inf_min)


    dataset = EmbeddingDataset(data, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)

    # Get output Distribution
    predictions = predict_with_uncertainty_batched(model, dataloader, n_iter=30)
    predictions = predictions.numpy()

    # Calculate entropy
    a, prob_mat = uncertainty_matrices(predictions)
    t, e, a = entropy_uncertainty(prob_mat)
    # t_min, t_max = np.min(t), np.max(t)
    #
    # entropy_norm = (t - t_min) / (t_max - t_min)

    return idxs,  t*influential_values#inf_norm*entropy_norm


def misclassifications_uncertainty_baseline(data,model,labels,preds):


    dataset = EmbeddingDataset(data, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)

    # Get output Distribution
    predictions = predict_with_uncertainty_batched(model, dataloader, n_iter=30)
    predictions = predictions.numpy()

    # Calculate entropy
    a, prob_mat = uncertainty_matrices(predictions)
    t, e, a = entropy_uncertainty(prob_mat)

    misclassified_labels_idxs = np.where(preds != labels)[0]
    unc = t[misclassified_labels_idxs]
    classified_labels_idxs = misclassified_labels_idxs

    return classified_labels_idxs, unc

def misclassifications_influence_baseline(data,labels,preds,mesh_idx):
    misclassified_labels_idxs = np.where(preds != labels)[0]


    # Randomly select indices to accept recommendations
    with open('/home/iroberts/projects/VisMeshSegmentation/influence_scores1.json', 'r') as f:
        influence_scores = json.load(f)

    influence_dict = {entry['mesh_idx']: entry['influence_per_vertex'] for entry in influence_scores}


    influential_mismatch_values = np.abs(np.array(influence_dict[mesh_idx])[misclassified_labels_idxs])


    return misclassified_labels_idxs,influential_mismatch_values

def deepview_variants(pred_wrapper,data,labels,model,mesh_idx):
    # --- Deep View Parameters ----
    batch_size = 32
    max_samples = 100000
    data_shape = (96,)
    resolution = 100
    N = 10
    lam = 1
    cmap = 'tab10'
    # to make shure deepview.show is blocking,
    # disable interactive mode
    interactive = False
    title = 'Automatic Relabeling'

    deepview = DeepViewLabelRevisit(pred_wrapper, np.arange(2), max_samples, batch_size, data_shape,
                                    N, lam, resolution, cmap, interactive, title, disc_dist=False)

    deepview.add_samples(data, labels)
    change_labels = deepview.recommend_label_correction(100)

    dataset = EmbeddingDataset(data, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)

    # Get output Distribution
    predictions = predict_with_uncertainty_batched(model, dataloader, n_iter=30)
    predictions = predictions.numpy()

    # Calculate entropy
    a, prob_mat = uncertainty_matrices(predictions)
    t, e, a = entropy_uncertainty(prob_mat)

    unc = t[change_labels]

    # Randomly select indices to accept recommendations
    with open('/home/iroberts/projects/VisMeshSegmentation/influence_scores.json', 'r') as f:
        influence_scores = json.load(f)

    influence_dict = {entry['mesh_idx']: entry['influence_per_vertex'] for entry in influence_scores}

    inf = np.abs(np.array(influence_dict[mesh_idx])[change_labels])

    return change_labels, unc, inf


def deepview_kmeans(pred_wrapper,data,labels,preds,model):
    # --- Deep View Parameters ----
    batch_size = 32
    max_samples = 100000
    data_shape = (96,)
    resolution = 100
    N = 10
    lam = 1
    cmap = 'tab10'
    # to make shure deepview.show is blocking,
    # disable interactive mode
    interactive = False
    title = 'Automatic Relabeling'

    deepview = DeepViewLabelRevisit(pred_wrapper, np.arange(2), max_samples, batch_size, data_shape,
                                    N, lam, resolution, cmap, interactive, title, disc_dist=False)

    deepview.add_samples(data, labels)
    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(deepview.embedded)
    changed_labels = labels.copy()
    for val in np.arange(2):
        idxs = np.where(kmeans.labels_ == val)
        avg_pred = int(np.mean(preds[idxs]))
        changed_labels[idxs] = avg_pred

    dataset = EmbeddingDataset(data, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)

    # Get output Distribution
    predictions = predict_with_uncertainty_batched(model, dataloader, n_iter=30)
    predictions = predictions.numpy()

    # Calculate entropy
    a, prob_mat = uncertainty_matrices(predictions)
    t, e, a = entropy_uncertainty(prob_mat)

    return changed_labels, t

def deepview_influence(data,influence,points_to_include):
    # --- Deep View Parameters ----
    batch_size = 32
    max_samples = 100000
    data_shape = (96,)
    resolution = 100
    N = 10
    lam = 1
    cmap = 'tab10'
    # to make shure deepview.show is blocking,
    # disable interactive mode
    interactive = False
    title = 'Automatic Relabeling'

    deepview = DeepViewLabelRevisit(pred_wrapper, np.arange(2), max_samples, batch_size, data_shape,
                                    N, lam, resolution, cmap, interactive, title, disc_dist=False)

    deepview.add_samples(data, labels)
    change_labels = deepview.recommend_label_correction(100)

    dataset = EmbeddingDataset(data, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)

    # Get output Distribution
    predictions = predict_with_uncertainty_batched(model, dataloader, n_iter=30)
    predictions = predictions.numpy()

    # Calculate entropy
    a, prob_mat = uncertainty_matrices(predictions)
    t, e, a = entropy_uncertainty(prob_mat)

    unc = t[change_labels]
    pass





