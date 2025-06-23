from improve_mesh_segmentation.partnet_grasp.dataset import PartNetGraspDataset

from tqdm import tqdm

import numpy as np
import pandas as pd
import json


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


def uniquify_corrections(corrections, noisy_labels, filename=None):
    """Filters corrections to the latest corrections and checks that they actually change a label."""
    unique_cors = []
    for idx, cor_suggestion in tqdm(enumerate(corrections), postfix="Uniquify corrections..."):
        # Get last/latest correction suggestion
        cor_suggestion = corrections[(cor_suggestion[:2] == corrections[:, :2]).all(axis=-1)][-1]

        # Check if the correction suggestion is already in the list of unique corrections
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


def compute_correction_precision_and_recall_and_f1(noisy_labels, corrections, gt_corrections):
    """Computes correction precision and recall of suggested corrections."""
    # Compute amount of true positives: Amount of correction suggestions that are correct
    true_positives = compute_tps(noisy_labels, corrections, gt_corrections)

    precision = true_positives / corrections.shape[0]
    recall = true_positives / gt_corrections.shape[0]
    f1 = 2 / (1/precision + 1/recall)

    # Compute amount of corrections to be made
    return precision, recall, f1


def evaluated_correction_prf_wrapper(noisy_data_path, expert_corrections_path, correction_files, result_filename):
    """Wraps the computation of correction precision, -recall and -f1 score for multiple experiments.

    Parameters
    ----------
    noisy_data_path: str
        The path to the noisy dataset.
    expert_corrections_path: str
        The path to the array that contains the expert (human/DeepView) corrections in form of triples:
        (mesh_idx, vertex_idx, label)
    correction_files: list
        A list of file-paths that point to *.npy-files that contain corrections in form of tuples:
        (mesh_idx, vertex_idx)
    result_filename: str
        The name of the resulting *.json-file into which the evaluation measures are stored.
    """
    original_triples = return_mesh_vertex_label_triples(data_path=noisy_data_path)
    expert_corrections = uniquify_corrections(
        corrections=np.loadtxt(expert_corrections_path, delimiter=",", dtype=np.int32), noisy_labels=original_triples
    )

    result_dict = {}
    for file_path in correction_files:
        # Load all corrections: (n_corrections, 2)
        if file_path[-3:] == "csv":
            method_corrections = pd.read_csv(file_path, header=None).to_numpy()
        elif file_path[-3:] == "npy":
            method_corrections = np.load(file_path)
        else:
            raise ValueError(
                f"File {file_path} is not a valid correction file. Only *.npy and *.csv files are supported."
            )
        if method_corrections.shape[1] > 2:
            # Check unique corrections
            print(f"\nBefore uniquification, method corrections shape: {method_corrections.shape}")
            method_corrections = uniquify_corrections(
                corrections=method_corrections, noisy_labels=original_triples
            )
            print(f"\nAfter uniquification, method corrections shape: {method_corrections.shape}")
            # If the method corrections contain labels, remove them
            method_corrections = method_corrections[:, :2]

        precision, recall, f1 = compute_correction_precision_and_recall_and_f1(
            noisy_labels=original_triples,
            corrections=method_corrections,
            gt_corrections=expert_corrections
        )
        result_dict[f"{file_path}"] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "amount_corrections": method_corrections.shape[0]
        }
        with open(result_filename, "w") as f:
            json.dump(result_dict, f, indent=4)
        print("\n", file_path, precision, recall, f1)
