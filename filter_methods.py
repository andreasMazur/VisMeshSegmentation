import json

import numpy as np
from sklearn.cluster import KMeans,DBSCAN
from sklvq import GLVQ

from helper_functions import *
from relabel_helpers.deepview_label_corrections import DeepViewLabelRevisit
from relabel_helpers.recommendation_functions import recommend_KNN_based


from scipy.spatial.distance import cdist
from collections import Counter


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
    # t_min, t_max = np.min(knn_30_corrections), np.max(knn_30_corrections)
    #
    # entropy_norm = (knn_30_corrections - t_min) / (t_max - t_min)

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


def knn_label_correction(data,labels,preds,combined_mesh_idxs,unc_dict,k):
    changed_labels, all_indices, keep_indices = recommend_KNN_based(data, labels, preds, k, recommendation_percentage=100)

    result = {}
    new_labels = {}
    for key, values in combined_mesh_idxs.items():
        positions = [i for i, v in enumerate(values) if v in keep_indices]
        new_labels[key] = changed_labels[values]
        if positions:
            result[key] = positions

    changed_idx_unc = {}
    for key in result.keys():
        changed_idx_unc[key] = unc_dict[key][result[key]]


    return result, changed_idx_unc,new_labels

def kmeans_label_correction(data,labels,preds,combined_mesh_idxs,unc_dict):
    kmeans = KMeans(n_clusters=5, random_state=0, n_init="auto").fit(data)
    changed_labels = labels.copy()

    for val in np.unique(kmeans.labels_):
        idxs = np.where(kmeans.labels_ == val)
        avg_pred = np.round(np.mean(preds[idxs]),0)
        changed_labels[idxs] = avg_pred

    changed_indices = np.where(changed_labels != labels)[0]


    result = {}
    new_labels ={}
    for key, values in combined_mesh_idxs.items():
        positions = [i for i, v in enumerate(values) if v in changed_indices]
        new_labels[key] = changed_labels[values]
        if positions:
            result[key] = positions

    changed_idx_unc = {}
    for key in result.keys():
        changed_idx_unc[key] = unc_dict[key][result[key]]


    return result, changed_idx_unc,new_labels

def dbscan_label_correction(data,labels,preds,combined_mesh_idxs,unc_dict):
    clustering = DBSCAN(min_samples=200).fit(data)
    changed_labels = labels.copy()

    for val in np.unique(clustering.labels_):
        idxs = np.where(clustering.labels_ == val)
        avg_pred = np.round(np.mean(preds[idxs]),0)
        changed_labels[idxs] = avg_pred

    changed_indices = np.where(changed_labels != labels)[0]


    result = {}
    new_labels ={}
    for key, values in combined_mesh_idxs.items():
        positions = [i for i, v in enumerate(values) if v in changed_indices]
        new_labels[key] = changed_labels[values]
        if positions:
            result[key] = positions

    changed_idx_unc = {}
    for key in result.keys():
        changed_idx_unc[key] = unc_dict[key][result[key]]


    return result, changed_idx_unc,new_labels


def lvq_label_correction(data, labels, preds, combined_mesh_idxs, unc_dict, prototype_n_per_class=5):
    lvq = GLVQ(
        distance_type="squared-euclidean",
        activation_type="swish",
        activation_params={"beta": 2},
        solver_type="steepest-gradient-descent",
        solver_params={"max_runs": 20, "step_size": 0.1},
        prototype_n_per_class=prototype_n_per_class,
    )
    # Train the model using the iris dataset
    lvq.fit(data, labels)

    # Predict the labels using the trained model
    predicted_labels = np.argmax(lvq.predict_proba(data), axis=1)

    changed_indices = np.where(predicted_labels != labels)[0]

    result = {}
    new_labels = {}
    for key, values in combined_mesh_idxs.items():
        positions = [i for i, v in enumerate(values) if v in changed_indices]
        new_labels[key] = predicted_labels[values]
        result[key] = positions

    changed_idx_unc = {}
    for key in result.keys():
        changed_idx_unc[key] = unc_dict[key][result[key]]

    return result, changed_idx_unc, new_labels

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
    kmeans = KMeans(n_clusters=5, random_state=0, n_init="auto").fit(deepview.embedded)
    # preds = kmeans.predict(deepview.embedded)
    changed_labels = labels.copy()
    for val in np.unique(kmeans.labels_):
        idxs = np.where(kmeans.labels_ == val)
        avg_pred = np.round(np.mean(preds[idxs]),0)
        changed_labels[idxs] = avg_pred

    changed_indices = np.where(changed_labels != labels)[0]

    dataset = EmbeddingDataset(data, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)

    # Get output Distribution
    predictions = predict_with_uncertainty_batched(model, dataloader, n_iter=30)
    predictions = predictions.numpy()

    # Calculate entropy
    a, prob_mat = uncertainty_matrices(predictions)
    t, e, a = entropy_uncertainty(prob_mat)

    return changed_indices, t[changed_indices]


def deepview_dbscan(pred_wrapper,data,labels,preds,model):
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
    clustering = DBSCAN().fit(data)
    # preds = kmeans.predict(deepview.embedded)
    changed_labels = labels.copy()
    for val in np.unique(clustering.labels_):
        idxs = np.where(clustering.labels_ == val)
        avg_pred = np.round(np.mean(preds[idxs]),0)
        changed_labels[idxs] = avg_pred

    changed_indices = np.where(changed_labels != labels)[0]

    dataset = EmbeddingDataset(data, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)

    # Get output Distribution
    predictions = predict_with_uncertainty_batched(model, dataloader, n_iter=30)
    predictions = predictions.numpy()

    # Calculate entropy
    a, prob_mat = uncertainty_matrices(predictions)
    t, e, a = entropy_uncertainty(prob_mat)

    return changed_indices, t[changed_indices]


def _lvq(data, labels, preds, model):
    # The creation of the model object used to fit the data to.
    lvq = GLVQ(
        distance_type="squared-euclidean",
        activation_type="swish",
        activation_params={"beta": 2},
        solver_type="steepest-gradient-descent",
        solver_params={"max_runs": 20, "step_size": 0.1},
        prototype_n_per_class=3,
    )
    # Train the model using the iris dataset
    lvq.fit(data, preds)

    # Predict the labels using the trained model
    predicted_labels = np.argmax(lvq.predict_proba(data),axis=1)

    changed_indices = np.where(predicted_labels != labels)[0]

    dataset = EmbeddingDataset(data, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)

    # Get output Distribution
    predictions = predict_with_uncertainty_batched(model, dataloader, n_iter=30)
    predictions = predictions.numpy()

    # Calculate entropy
    a, prob_mat = uncertainty_matrices(predictions)
    t, e, a = entropy_uncertainty(prob_mat)

    return changed_indices,


def fit_classwise_kmeans(data, labels, n_clusters=5):
    class_to_centroids = {}
    for label in np.unique(labels):
        idxs = np.where(labels == label)[0]
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(data[idxs])
        class_to_centroids[label] = kmeans.cluster_centers_
    return class_to_centroids


def predict_with_prototypes(
        data,
        class_to_centroids,
        top_k=1,
        voting="majority"  # or "weighted"
):
    """
    Predicts class labels based on nearest prototypes.

    Args:
        data: (N, D) array of points to classify.
        class_to_centroids: dict {label: (K, D) array}
        top_k: number of nearest prototypes to consider
        voting: 'majority' or 'weighted' voting scheme

    Returns:
        preds: (N,) array of predicted labels
    """
    # Stack all centroids with labels
    all_centroids = []
    centroid_labels = []
    for label, centroids in class_to_centroids.items():
        all_centroids.append(centroids)
        centroid_labels.extend([label] * len(centroids))
    all_centroids = np.vstack(all_centroids)  # (C_total, D)
    centroid_labels = np.array(centroid_labels)  # (C_total,)

    preds = []
    for x in data:
        dists = cdist([x], all_centroids)[0]  # (C_total,)
        nearest_idxs = np.argsort(dists)[:top_k]
        nearest_labels = centroid_labels[nearest_idxs]

        if voting == "majority":
            pred_label = Counter(nearest_labels).most_common(1)[0][0]
        elif voting == "weighted":
            weights = 1 / (dists[nearest_idxs] + 1e-8)
            label_weights = {}
            for lbl, w in zip(nearest_labels, weights):
                label_weights[lbl] = label_weights.get(lbl, 0) + w
            pred_label = max(label_weights.items(), key=lambda x: x[1])[0]
        else:
            raise ValueError("Voting must be 'majority' or 'weighted'")

        preds.append(pred_label)

    return np.array(preds)


def supervised_kmeans_label_correction(data, labels, pred, combined_mesh_idxs, unc_dict, n_clusters=5, topk=1):
    class_to_centroids = fit_classwise_kmeans(data, labels, n_clusters=n_clusters)

    preds = predict_with_prototypes(data, class_to_centroids, top_k=topk, voting="majority" )

    changed_indices = np.where(preds != labels)[0]

    result = {}
    new_labels = {}
    for key, values in combined_mesh_idxs.items():
        positions = [i for i, v in enumerate(values) if v in changed_indices]
        new_labels[key] = preds[values]
        result[key] = positions

    changed_idx_unc = {}
    for key in result.keys():
        changed_idx_unc[key] = unc_dict[key][result[key]]

    return result, changed_idx_unc, new_labels






