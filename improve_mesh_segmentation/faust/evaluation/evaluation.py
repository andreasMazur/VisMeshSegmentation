from improve_mesh_segmentation.faust.segmentation_data.faust_segmentation_dataset import FaustSegmentationDataset

from tqdm import tqdm

import torch
import numpy as np
import os


def get_true_positives(quintuples):
    # 1.) Filter down to vertices that were corrected
    set_of_all_corrections = quintuples[quintuples[:, 4] == 1]

    # 2.) Filter down to vertices that were noisy
    set_of_correct_corrections = set_of_all_corrections[set_of_all_corrections[:, 2] != set_of_all_corrections[:, 3]]

    return set_of_correct_corrections.shape[0], set_of_all_corrections.shape[0]


def compute_correction_statistics(dataset_path, segmentation_labels_path, logging_dir, corrections_path):
    # Get noisy labels
    noisy_dataset = FaustSegmentationDataset(
        path_to_zip=dataset_path,
        path_to_segmentation_labels=segmentation_labels_path,
        logging_dir=logging_dir,
        set_type=3,
        only_signal=False,
        device=None,
        noise_level=None,
        set_indices=None
    )

    # Create logging dir with clean segmentation labels
    clean_logging_dir = f"{os.path.dirname(logging_dir)}/clean_faust_segmentation"
    os.makedirs(clean_logging_dir, exist_ok=True)
    clean_segmentation_labels = np.load(segmentation_labels_path)
    np.save(f"{clean_logging_dir}/noisy_segmentation_labels.npy", np.tile(clean_segmentation_labels, (100, 1)))

    # Get clean labels
    clean_dataset = FaustSegmentationDataset(
        path_to_zip=dataset_path,
        path_to_segmentation_labels=segmentation_labels_path,
        logging_dir=clean_logging_dir,  # No noise added
        set_type=3,
        only_signal=False,
        device=None,
        noise_level=None,
        set_indices=None,
        put_into_correct_order=True
    )

    # load corrections
    corrections = np.load(corrections_path)

    tps_counter = 0
    corrections_counter = 0
    to_be_found_counter = 0
    for mesh_idx, ((_, gt_o), (_, gt_c)) in tqdm(enumerate(zip(noisy_dataset, clean_dataset))):
        mesh_indices = torch.full((gt_o.shape[0],), fill_value=mesh_idx)
        vertex_indices = torch.arange(gt_o.shape[0])

        to_be_corrected = np.zeros((gt_o.shape[0]))
        to_be_corrected[corrections[corrections[:, 0] == mesh_idx][:, 1]] = 1
        to_be_corrected = torch.tensor(to_be_corrected)

        # 0: mesh index
        # 1: vertex index
        # 2: observed label
        # 3: clean label
        # 4: to be corrected (1) or not (0) according to method
        quintuples = torch.stack([mesh_indices, vertex_indices, gt_o, gt_c, to_be_corrected], dim=-1)
        n_true_positives, n_all_corrections = get_true_positives(quintuples)
        n_to_be_found = quintuples[quintuples[:, 2] != quintuples[:, 3]].shape[0]  # all noisy vertices

        tps_counter += n_true_positives
        corrections_counter += n_all_corrections
        to_be_found_counter += n_to_be_found

    noise_precision = tps_counter / corrections_counter
    noise_recall = tps_counter / to_be_found_counter
    noise_f1 = 2 / ((1 / noise_recall) + (1 / noise_precision))
    print(f"Noise Precision: {noise_precision}, Noise Recall: {noise_recall}, Noise F1: {noise_f1}")
    return noise_precision, noise_recall, noise_f1
