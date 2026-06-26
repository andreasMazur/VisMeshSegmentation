import numpy as np
import torch
import umap
from collections import Counter
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklvq import GLVQ

from improve_mesh_segmentation.comparison_methods.helpers import embedding_entropy_uncertainty
from improve_mesh_segmentation.comparison_methods.recommendation import recommend_knn_binarized


def _embedding_array(data):
    if torch.is_tensor(data):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def _knn(data, labels, preds, model, n_neighbors):
    changed_labels, changed_indices = recommend_knn_binarized(
        data, labels, preds, n_neighbors=n_neighbors, recommendation_percentage=100,
    )

    t = embedding_entropy_uncertainty(data, labels, model)
    return changed_indices, t[changed_indices], changed_labels


def umap_knn(pred_wrapper, data, labels, preds, model, method_parameters):
    mapper = umap.UMAP(
        n_neighbors=method_parameters[1],
        n_components=method_parameters[2],
        metric='euclidean',
    ).fit_transform(_embedding_array(data))

    changed_labels, changed_indices = recommend_knn_binarized(
        mapper, labels, preds, n_neighbors=method_parameters[0], recommendation_percentage=100,
    )

    t = embedding_entropy_uncertainty(data, labels, model)
    return changed_indices, t[changed_indices], changed_labels


def knn_label_correction(data, labels, preds, combined_mesh_idxs, unc_dict, k):
    changed_labels, changed_indices = recommend_knn_binarized(
        data, labels, preds, n_neighbors=k, recommendation_percentage=100,
    )

    result = {}
    new_labels = {}
    for key, values in combined_mesh_idxs.items():
        positions = [i for i, v in enumerate(values) if v in changed_indices]
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
        values, counts = np.unique(preds[idxs], return_counts=True)
        changed_labels[idxs] = values[np.argmax(counts)]

    changed_indices = np.where(changed_labels != labels)[0]
    final_labs = np.zeros_like(labels, dtype=int)
    final_labs[changed_indices] = 1

    result = {}
    new_labels = {}
    for key, values in combined_mesh_idxs.items():
        positions = [i for i, v in enumerate(values) if v in changed_indices]
        new_labels[key] = final_labs[values]
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

    final_labs = np.zeros_like(labels, dtype=int)
    final_labs[changed_indices] = 1

    result = {}
    new_labels = {}
    for key, values in combined_mesh_idxs.items():
        positions = [i for i, v in enumerate(values) if v in changed_indices]
        new_labels[key] = final_labs[values]
        result[key] = positions

    changed_idx_unc = {key: unc_dict[key][result[key]] for key in result.keys()}
    return result, changed_idx_unc, new_labels


def _kmeans(data, labels, preds, model, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto").fit(data)
    changed_labels = labels.copy()

    for val in np.unique(kmeans.labels_):
        idxs = np.where(kmeans.labels_ == val)
        values, counts = np.unique(preds[idxs], return_counts=True)
        changed_labels[idxs] = values[np.argmax(counts)]

    changed_indices = np.where(changed_labels != labels)[0]
    final_labs = np.zeros_like(labels, dtype=int)
    final_labs[changed_indices] = 1

    t = embedding_entropy_uncertainty(data, labels, model)
    return changed_indices, t[changed_indices], final_labs


def umap_kmeans(pred_wrapper, data, labels, preds, model, method_parameters):
    mapper = umap.UMAP(
        n_neighbors=method_parameters[1],
        n_components=method_parameters[2],
        metric='euclidean',
    ).fit_transform(_embedding_array(data))

    kmeans = KMeans(n_clusters=method_parameters[0], random_state=0, n_init="auto").fit(mapper)
    changed_labels = labels.copy()

    for val in np.unique(kmeans.labels_):
        idxs = np.where(kmeans.labels_ == val)
        values, counts = np.unique(preds[idxs], return_counts=True)
        changed_labels[idxs] = values[np.argmax(counts)]

    changed_indices = np.where(changed_labels != labels)[0]
    final_labs = np.zeros_like(labels, dtype=int)
    final_labs[changed_indices] = 1

    t = embedding_entropy_uncertainty(data, labels, model)
    return changed_indices, t[changed_indices], final_labs


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
        pred_label = Counter(nearest_labels).most_common(1)[0][0]
        preds.append(pred_label)
    return np.array(preds)


def supervised_kmeans_label_correction(data, labels, combined_mesh_idxs, unc_dict, n_clusters=3, topk=1):
    class_to_centroids = fit_classwise_kmeans(data, labels, n_clusters=n_clusters)
    preds = predict_with_prototypes(data, class_to_centroids, top_k=topk, voting="majority")
    changed_indices = np.where(preds != labels)[0]

    final_labs = np.zeros_like(labels, dtype=int)
    final_labs[changed_indices] = 1

    result = {}
    new_labels = {}
    for key, values in combined_mesh_idxs.items():
        positions = [i for i, v in enumerate(values) if v in changed_indices]
        new_labels[key] = final_labs[values]
        result[key] = positions

    changed_idx_unc = {key: unc_dict[key][result[key]] for key in result.keys()}
    return result, changed_idx_unc, new_labels
