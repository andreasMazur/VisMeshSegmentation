from improve_mesh_segmentation.partnet_grasp.dataset import PartNetGraspDataset

from tqdm import tqdm

import numpy as np


def return_mesh_vertex_label_triples(data_path):
    """Returns all mesh-vertex-label triples."""
    dataset = PartNetGraspDataset(set_type=3, path_to_zip=data_path)
    mesh_vertex_label_triples = []
    for mesh_idx, ((signal, bc), gt) in enumerate(dataset):
        vertex_indices = np.arange(signal.shape[0])
        mesh_vertex_label_triples.append(
            np.concatenate(
                [
                    np.full_like(vertex_indices, mesh_idx).reshape(-1, 1),
                    vertex_indices.reshape(-1, 1),
                    gt.numpy().reshape(-1, 1)
                ],
                axis=-1
            )
        )
    return np.concatenate(mesh_vertex_label_triples, axis=0)


def load_dv_corrections(path_to_dv_corrections):
    """Loads the deepview corrections."""
    return np.loadtxt(path_to_dv_corrections, delimiter=",", dtype=np.int32)


def filter_deepview_corrections(corrections, noisy_labels, filename=None):
    """Filters deepview corrections to the latest corrections."""
    unique_cors = []
    for idx, cor_suggestion in tqdm(enumerate(corrections), postfix="Filtering DV corrections..."):
        # Get last/latest correction suggestion
        cor_suggestion = corrections[(cor_suggestion[:2] == corrections[:, :2]).all(axis=-1)][-1]
        if len(unique_cors) > 0 and (cor_suggestion[:2] == np.array(unique_cors)[:, :2]).all(axis=-1).any():
            continue

        # Only add corrections that actually change the label
        noisy_label_idx = np.where((cor_suggestion[:2] == noisy_labels[:, :2]).all(axis=-1))[0][0]
        if noisy_labels[noisy_label_idx][-1] != cor_suggestion[-1]:
            unique_cors.append(cor_suggestion)

    # Store unique deepview corrections
    unique_cors = np.array(unique_cors)
    if filename is not None:
        np.save(filename, unique_cors)

    return unique_cors


def compute_tps(noisy_labels, corrections, gt_corrections):
    """Compute amount of true positive labels: Corrections that change actually invalid labels."""
    true_positives = 0
    for cor_suggestion in tqdm(corrections, postfix="Counting correct corrections..."):
        # Get noisy vertex that shall be flipped according to 'cor_suggestion'
        noisy_vertex = noisy_labels[(cor_suggestion == noisy_labels[:, :2]).all(axis=-1)][0]

        # Increment true positives in case gt-correction changes it too
        gt_correction = gt_corrections[(noisy_vertex[:2] == gt_corrections[:, :2]).all(axis=-1)]
        if gt_correction.shape[0] > 0 and gt_correction[0, -1] != noisy_vertex[-1]:
            true_positives += 1
    return true_positives


def compute_correction_precision_and_recall(noisy_labels, corrections, gt_corrections):
    """Computes correction precision and recall of suggested corrections."""
    # Compute amount of true positives: Amount of correction suggestions that are correct
    true_positives = compute_tps(noisy_labels, corrections, gt_corrections)

    precision = true_positives / corrections.shape[0]
    recall = true_positives / gt_corrections.shape[0]

    # Compute amount of corrections to be made
    return precision, recall
