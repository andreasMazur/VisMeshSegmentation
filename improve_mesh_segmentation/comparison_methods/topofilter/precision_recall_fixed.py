from dataset import PartNetGraspDataset
from pathlib import Path
import pandas as pd
import numpy as np
import json
from termcolor import cprint

from rich.progress import Progress, track
from IPython import embed

from multiprocessing.pool import ThreadPool
from collections import deque
import time

def filter_deepview_corrections(corrections, noisy_labels, filename=None):
    """Filters deepview corrections to the latest corrections."""
    unique_cors = []
    # for idx, cor_suggestion in track(enumerate(corrections), description="Filtering DV corrections..."):
    for idx, cor_suggestion in enumerate(corrections):
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
    # for cor_suggestion in track(corrections, description="Counting correct corrections..."):
    for i, cor_suggestion in enumerate(corrections):
        # Get noisy vertex that shall be flipped according to 'cor_suggestion'
        noisy_vertex = noisy_labels[(cor_suggestion[:2] == noisy_labels[:, :2]).all(axis=-1)][0]
        assert noisy_vertex[-1] != cor_suggestion[-1], "Noisy vertex and correction suggestion must differ in label!"

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

def evaluate_all_prc(
    path_to_data,
    result_filename,
):

    # cprint("Creating thread pool and queue ...", "blue")
    # pool = ThreadPool(n_workers)
    # pending = deque()

    #cprint("Reading manual corrections ...", "blue")
    #dv_corr_path = Path(__file__).parent.parent / 'improve_mesh_segmentation' / 'data_correction' / 'partnet_correction.csv'
    dv_corr_path = Path('unq_dv_triplets.npy')
    #dv_csv = pd.read_csv(dv_corr_path.as_posix(), header=None, index_col=None)
    #dv_triplets = dv_csv.to_numpy()[:, [0, 1, 2]]

    cprint("Loading full datasets ...", "blue")
    dataset = PartNetGraspDataset(
        path_to_zip=path_to_data,
        correction_file_path=dv_corr_path,
        set_type=3,
        for_adapt=False,
    )

    cprint("Creating original triplets ...", "blue")
    curr_shape_idx = 0
    shape_ids = []
    local_vert_ids = []
    labels = []
    for shape_idx, global_vert_idxs in enumerate(dataset.shape_indices):
        shape_ids.extend([shape_idx] * len(global_vert_idxs))
        local_vert_ids.extend(global_vert_idxs-curr_shape_idx)
        labels.extend(dataset.labels[global_vert_idxs])
        curr_shape_idx += len(global_vert_idxs)
    original_triplets = np.array([shape_ids, local_vert_ids, labels]).T

    cprint("Filtering deepview corrections ...", "blue")
    #unq_dv_triplets = filter_deepview_corrections(dv_triplets, original_triplets)
    unq_dv_triplets  = np.load('unq_dv_triplets.npy')

    cprint("Loading train dataset ...", "blue")
    dataset = PartNetGraspDataset(
        path_to_zip=path_to_data,
        correction_file_path=dv_corr_path,
        set_type=0,
        for_adapt=False,
    )

    cprint("Creating original triplets for train dataset ...", "blue")
    curr_shape_idx = 0
    shape_ids = []
    local_vert_ids = []
    labels = []
    for shape_idx, global_vert_idxs in enumerate(dataset.shape_indices):
        shape_ids.extend([shape_idx] * len(global_vert_idxs))
        local_vert_ids.extend(global_vert_idxs-curr_shape_idx)
        labels.extend(dataset.labels[global_vert_idxs])
        curr_shape_idx += len(global_vert_idxs)
    original_triplets = np.array([shape_ids, local_vert_ids, labels]).T

    results_dict = {}
    auto_corrections_path = Path(__file__).parent / "fixed_clearing/results"
    auto_corrections_files = auto_corrections_path.glob("*.csv")
    best_f1 = 0.0
    for auto_corrections_file in track(auto_corrections_files, "Evaluating correction files", total=len(list(auto_corrections_path.glob("*.csv")))):
        csv = pd.read_csv(auto_corrections_file.as_posix(), header=None, index_col=None)
        data = csv.to_numpy()[:, [0, 1, 2]]
        precision, recall, f1 = compute_correction_precision_and_recall_and_f1(original_triplets, data, unq_dv_triplets)
        if f1 > best_f1:
            best_f1 = f1
            cprint(f"New best f1:\n{auto_corrections_file.name}\t- Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}", "green")

        results_dict[auto_corrections_file.name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

        with open(result_filename, 'w') as f:
            json.dump(results_dict, f, indent=4)

    cprint("Evaluation completed. Results saved to {}".format(result_filename), "blue")
    # Get top 3 f1 results
    sorted_results = sorted(results_dict.items(), key=lambda x: x[1]['f1'], reverse=True)
    cprint("Top 10 results:", "red")
    for name, metrics in sorted_results[:3]:
        cprint(f" * {name}:\n\tPrecision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}", "green")

    cprint("\n\n Top 3 PointNet models:", "yellow")
    for name, metrics in [res for res in sorted_results if res[0].startswith('corrections_pointnet')][:3]:
        cprint(f" * {name}:\n\tnum_corrections: {metrics['num_corrections']} Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}", "yellow")

    # with Progress("Evaluating correction files") as progress:
        # task = progress.add_task("[blue]Evaluating...", total=len(list(auto_corrections_path.glob("*.csv"))))
        # while True:
        #     try:
        #         while len(pending) > 0 and pending[0].ready():
        #             result = pending.popleft().get()
        #             results_dict[result[0]] = {
        #                 'precision': result[1],
        #                 'recall': result[2],
        #                 'f1': result[3]
        #             }
        #             if result[3] > best_f1:
        #                 cprint(f"New best f1:\n{result[0]}\t- Precision: {result[1]:.4f}, Recall: {result[2]:.4f}, F1: {result[3]:.4f}", "green")
        #             with open(result_filename, 'w') as f:
        #                 json.dump(results_dict, f, indent=4)
        #             progress.update(task, advance=1)
        #         if len(pending) < n_workers:
        #             auto_corrections_file = next(auto_corrections_files)
        #             csv = pd.read_csv(auto_corrections_file.as_posix(), header=None, index_col=None)
        #             data = csv.to_numpy()[:, [0, 1, 2]]
        #             pending.append(pool.apply_async(compute_correction_precision_and_recall_and_f1, (auto_corrections_file.name, original_triplets, data, unq_dv_triplets)))
        #             print("appended new task to queue:", auto_corrections_file.name)
        #     except StopIteration:
        #         while len(pending) > 0:
        #             if pending[0].ready():
        #                 result = pending.popleft().get()
        #                 results_dict[result[0]] = {
        #                     'precision': result[1],
        #                     'recall': result[2],
        #                     'f1': result[3]
        #                 }
        #                 if result[3] > best_f1:
        #                     cprint(f"New best f1:\n{result[0]}\t- Precision: {result[1]:.4f}, Recall: {result[2]:.4f}, F1: {result[3]:.4f}", "green")
        #                 with open(result_filename, 'w') as f:
        #                     json.dump(results_dict, f, indent=4)
        #                 progress.update(task, advance=1)
        #             else:
        #                 time.sleep(0.1)
        #         break


if __name__ == "__main__":
    evaluate_all_prc(
        '../../SegLabelCorrection/data/partnet_grasp/partnet_grasp.zip',
        'fixed_clearing/topofilter_results_fixed.json'
    )
