import os

import numpy as np
import scipy as sp
import torch
from tqdm import tqdm

from improve_mesh_segmentation.comparison_methods import filter_methods_faust as faust_filters
from improve_mesh_segmentation.comparison_methods import filter_methods_partnet as partnet_filters
from improve_mesh_segmentation.comparison_methods.helpers import StochasticModel, global_mesh_sort, write_label_changes
from improve_mesh_segmentation.data_correction.correct_sub_partnet import embed


def _output_filename(method_name, method_parameter):
    if isinstance(method_parameter, tuple):
        suffix = "_".join(str(p) for p in method_parameter)
        return f"{method_name}_{suffix}.csv"
    return f"{method_name}_{method_parameter}.csv"


def _pred_wrapper(imcnn, device="cpu"):
    def wrapper(data):
        logits = imcnn.model.output_dense(torch.tensor(data).float().to(device))
        return sp.special.softmax(logits.detach().cpu().numpy(), axis=-1)
    return wrapper


def run_local_label_correction(
    method_name,
    filters,
    imcnn,
    classification_head,
    dataset,
    output_dir,
    method_parameter,
    device="cpu",
    use_umap=False,
):
    dropout_prob = 0.5
    stochastic_model = StochasticModel(classification_head, dropout_prob).to(device)
    pred_wrapper = _pred_wrapper(imcnn, device=device)

    mesh_indices = {}
    mesh_unc = {}
    mesh_labels = {}

    for mesh_idx, ((signal, bc), labels) in tqdm(enumerate(dataset), desc=method_name):
        if torch.is_tensor(labels):
            labels = labels.detach().cpu().numpy()
        else:
            labels = np.array(labels)

        embeddings = torch.tensor(embed(imcnn, [signal, bc]))
        with torch.no_grad():
            mesh_preds = classification_head(embeddings.to(device))
            mesh_preds = np.argmax(mesh_preds.detach().cpu().numpy(), axis=1)

        if use_umap:
            if method_name == "umap_knn_observed":
                idxs, unc, new_labels = filters.umap_knn(
                    pred_wrapper, embeddings, labels, labels, stochastic_model, method_parameter,
                )
            elif method_name == "umap_kmeans_observed":
                idxs, unc, new_labels = filters.umap_kmeans(
                    pred_wrapper, embeddings, labels, labels, stochastic_model, method_parameter,
                )
            else:
                raise NotImplementedError(f"Unknown UMAP method: {method_name}")
        elif method_name == "deepview_knn_observed":
            idxs, unc, new_labels = filters.deepview_knn(
                pred_wrapper, embeddings, labels, labels, stochastic_model, method_parameter,
            )
        elif method_name == "deepview_kmeans_observed":
            idxs, unc, new_labels = filters.deepview_kmeans(
                pred_wrapper, embeddings, labels, labels, stochastic_model, method_parameter,
            )
        elif method_name == "iterative_knn_observed":
            idxs, unc, new_labels = filters._knn(
                embeddings, labels, labels, stochastic_model, method_parameter,
            )
        elif method_name == "iterative_kmeans_observed":
            idxs, unc, new_labels = filters._kmeans(
                embeddings, labels, labels, stochastic_model, method_parameter,
            )
        else:
            raise NotImplementedError(f"Unknown local method: {method_name}")

        mesh_indices[mesh_idx] = idxs
        mesh_unc[mesh_idx] = unc
        mesh_labels[mesh_idx] = new_labels

    sorted_by_unc, _ = global_mesh_sort(mesh_indices, mesh_unc)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, _output_filename(method_name, method_parameter))
    write_label_changes(output_path, sorted_by_unc, mesh_labels)
    return output_path


def run_global_label_correction_partnet(
    method_name,
    embs,
    labels,
    preds,
    mesh_map,
    unc_dict,
    output_dir,
    method_parameter,
):
    filters = partnet_filters
    if method_name == "global_knn_observed":
        idxs, idx_unc, mesh_labels = filters.knn_label_correction(
            embs, labels, labels, mesh_map, unc_dict, method_parameter,
        )
    elif method_name == "global_kmeans_observed":
        idxs, idx_unc, mesh_labels = filters.kmeans_label_correction(
            embs, labels, labels, mesh_map, unc_dict, method_parameter,
        )
    elif method_name == "global_lvq_observed":
        idxs, idx_unc, mesh_labels = filters.lvq_label_correction(
            embs, labels, mesh_map, unc_dict, method_parameter,
        )
    elif method_name == "global_supervised_kmeans_observed":
        idxs, idx_unc, mesh_labels = filters.supervised_kmeans_label_correction(
            embs, labels, mesh_map, unc_dict, method_parameter,
        )
    else:
        raise NotImplementedError(f"Unknown global method: {method_name}")

    sorted_by_unc, _ = global_mesh_sort(idxs, idx_unc)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, _output_filename(method_name, method_parameter))
    write_label_changes(output_path, sorted_by_unc, mesh_labels)
    return output_path


def run_global_label_correction_faust(
    method_name,
    embs,
    labels,
    preds,
    mesh_map,
    unc_dict,
    output_dir,
    method_parameter,
):
    filters = faust_filters
    if method_name == "global_knn_observed":
        idxs, idx_unc, mesh_labels = filters.knn_label_correction(
            embs, labels, labels, mesh_map, unc_dict, method_parameter,
        )
    elif method_name == "global_kmeans_observed":
        idxs, idx_unc, mesh_labels = filters.kmeans_label_correction(
            embs, labels, labels, mesh_map, unc_dict, method_parameter,
        )
    elif method_name == "global_lvq_observed":
        idxs, idx_unc, mesh_labels = filters.lvq_label_correction(
            embs, labels, mesh_map, unc_dict, method_parameter,
        )
    elif method_name == "global_supervised_kmeans_observed":
        idxs, idx_unc, mesh_labels = filters.supervised_kmeans_label_correction(
            embs, labels, mesh_map, unc_dict, method_parameter,
        )
    else:
        raise NotImplementedError(f"Unknown global method: {method_name}")

    sorted_by_unc, _ = global_mesh_sort(idxs, idx_unc)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, _output_filename(method_name, method_parameter))
    write_label_changes(output_path, sorted_by_unc, mesh_labels)
    return output_path
