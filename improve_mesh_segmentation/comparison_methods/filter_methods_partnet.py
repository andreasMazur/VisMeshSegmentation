import numpy as np
import torch
from collections import Counter
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklvq import GLVQ

from improve_mesh_segmentation.comparison_methods.deepview_corrections import DeepViewLabelRevisit
from improve_mesh_segmentation.comparison_methods.helpers import embedding_entropy_uncertainty
from improve_mesh_segmentation.comparison_methods.recommendation import recommend_knn_based


def _knn(data, labels, preds, model, n_neighbors):
    changed_labels, _, keep_indices = recommend_knn_based(
        data, labels, preds, n_neighbors=n_neighbors, recommendation_percentage=100
    )
    changed_indices = np.where(changed_labels != labels)[0]

    t = embedding_entropy_uncertainty(data, labels, model)
    return changed_indices, t[changed_indices], changed_labels


def deepview_knn(pred_wrapper, data, labels, preds, model, k):
    batch_size = 32
    max_samples = 100000
    data_shape = (32,)
    resolution = 100
    deepview = DeepViewLabelRevisit(
        pred_wrapper, np.unique(labels), max_samples, batch_size, data_shape,
        10, 1, resolution, 'tab10', False, 'Automatic Relabeling', disc_dist=False,
    )
    deepview.add_samples(data, labels)

    changed_labels, _, change_indices = recommend_knn_based(
        deepview.embedded, labels, preds, k, recommendation_percentage=100,
    )

    t = embedding_entropy_uncertainty(data, labels, model)
    return change_indices, t[change_indices], changed_labels


def knn_label_correction(data, labels, preds, combined_mesh_idxs, unc_dict, k):
    changed_labels, _, keep_indices = recommend_knn_based(data, labels, preds, k, recommendation_percentage=100)

    result = {}
    new_labels = {}
    for key, values in combined_mesh_idxs.items():
        positions = [i for i, v in enumerate(values) if v in keep_indices]
        new_labels[key] = changed_labels[values]
        if positions:
            result[key] = positions

    changed_idx_unc = {key: unc_dict[key][result[key]] for key in result.keys()}
    return result, changed_idx_unc, new_labels


def kmeans_label_correction(data, labels, preds, combined_mesh_idxs, unc_dict, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto").fit(data)
    changed_labels = labels.copy()

    for val in np.unique(kmeans.labels_):
        idxs = np.where(kmeans.labels_ == val)
        avg_pred = np.round(np.mean(preds[idxs]), 0)
        changed_labels[idxs] = avg_pred

    changed_indices = np.where(changed_labels != labels)[0]
    result = {}
    new_labels = {}
    for key, values in combined_mesh_idxs.items():
        positions = [i for i, v in enumerate(values) if v in changed_indices]
        new_labels[key] = changed_labels[values]
        if positions:
            result[key] = positions

    changed_idx_unc = {key: unc_dict[key][result[key]] for key in result.keys()}
    return result, changed_idx_unc, new_labels


def lvq_label_correction(data, labels, combined_mesh_idxs, unc_dict, prototype_n_per_class=5):
    lvq = GLVQ(
        distance_type="squared-euclidean",
        activation_type="swish",
        activation_params={"beta": 2},
        solver_type="steepest-gradient-descent",
        solver_params={"max_runs": 20, "step_size": 0.1},
        prototype_n_per_class=prototype_n_per_class,
    )
    lvq.fit(data, labels)
    predicted_labels = np.argmax(lvq.predict_proba(data), axis=1)
    changed_indices = np.where(predicted_labels != labels)[0]

    result = {}
    new_labels = {}
    for key, values in combined_mesh_idxs.items():
        positions = [i for i, v in enumerate(values) if v in changed_indices]
        new_labels[key] = predicted_labels[values]
        result[key] = positions

    changed_idx_unc = {key: unc_dict[key][result[key]] for key in result.keys()}
    return result, changed_idx_unc, new_labels


def _kmeans(data, labels, preds, model, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto").fit(data)
    changed_labels = labels.copy()
    for val in np.unique(kmeans.labels_):
        idxs = np.where(kmeans.labels_ == val)
        avg_pred = np.round(np.mean(preds[idxs]), 0)
        changed_labels[idxs] = avg_pred

    changed_indices = np.where(changed_labels != labels)[0]
    t = embedding_entropy_uncertainty(data, labels, model)
    return changed_indices, t[changed_indices], changed_labels


def deepview_kmeans(pred_wrapper, data, labels, preds, model, num_clusters):
    batch_size = 32
    max_samples = 100000
    data_shape = (32,)
    resolution = 100
    deepview = DeepViewLabelRevisit(
        pred_wrapper, np.unique(labels), max_samples, batch_size, data_shape,
        10, 1, resolution, 'tab10', False, 'Automatic Relabeling', disc_dist=False,
    )
    deepview.add_samples(data, labels)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto").fit(deepview.embedded)
    changed_labels = labels.copy()
    for val in np.unique(kmeans.labels_):
        idxs = np.where(kmeans.labels_ == val)
        avg_pred = np.round(np.mean(preds[idxs]), 0)
        changed_labels[idxs] = avg_pred

    changed_indices = np.where(changed_labels != labels)[0]
    t = embedding_entropy_uncertainty(data, labels, model)
    return changed_indices, t[changed_indices], changed_labels


def fit_classwise_kmeans(data, labels, n_clusters=5):
    class_to_centroids = {}
    for label in np.unique(labels):
        idxs = np.where(labels == label)[0]
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(data[idxs])
        class_to_centroids[label] = kmeans.cluster_centers_
    return class_to_centroids


def predict_with_prototypes(data, class_to_centroids, top_k=1, voting="majority"):
    all_centroids = []
    centroid_labels = []
    for label, centroids in class_to_centroids.items():
        all_centroids.append(centroids)
        centroid_labels.extend([label] * len(centroids))
    all_centroids = np.vstack(all_centroids)
    centroid_labels = np.array(centroid_labels)

    preds = []
    for x in data:
        dists = cdist([x], all_centroids)[0]
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


def supervised_kmeans_label_correction(data, labels, combined_mesh_idxs, unc_dict, n_clusters=3, topk=1):
    class_to_centroids = fit_classwise_kmeans(data, labels, n_clusters=n_clusters)
    preds = predict_with_prototypes(data, class_to_centroids, top_k=topk, voting="majority")
    changed_indices = np.where(preds != labels)[0]

    result = {}
    new_labels = {}
    for key, values in combined_mesh_idxs.items():
        positions = [i for i, v in enumerate(values) if v in changed_indices]
        new_labels[key] = preds[values]
        result[key] = positions

    changed_idx_unc = {key: unc_dict[key][result[key]] for key in result.keys()}
    return result, changed_idx_unc, new_labels
