import os
from collections import Counter

import torch
from sklearn.model_selection import KFold
import json

from relabel_helpers.helper_functions import write_label_changes
from improve_mesh_segmentation.partnet_grasp.dataset import PartNetGraspDataset, processed_partnet_grasp_generator
from improve_mesh_segmentation.training.imcnn import SegImcnn
from run_through.step_3 import EXPERIMENT_DIRECTORY, PARTNET_GRASP
import numpy as np

""" Step 4: Train an initial IMCNN

    A trained IMCNN is required for the label correction process. By running this script, you train an IMCNN.
"""


def cv_model_pred(signal, bc, labels, voting="majority", percentage_misclass=0.5):
    """
       Args:
           signal, bc: Input mesh signal and barycentric coordinates
           labels: Ground truth labels for each vertex (numpy array)
           voting: 'majority', 'consensus', or 'percentage'
           percentage_misclass: float in [0, 1], threshold for 'percentage' voting

       Returns:
           final_preds: (num_points,) numpy array of predicted labels
           misclassified_idxs: list of indices based on voting strategy
       """
    # models_path = "/run_through/logs/cv/"
    # model_dirs = ['model_cv_0', 'model_cv_1', 'model_cv_2', 'model_cv_3', 'model_cv_4', 'model_cv_5', 'model_cv_6']

    models_path = "/home/iroberts/projects/VisMeshSegmentation/run_through/logs/behanvior_exp/cv/"
    model_dirs = ['model_cv_0', 'model_cv_1', 'model_cv_2', 'model_cv_3', 'model_cv_4', 'model_cv_5', 'model_cv_6','model_cv_7', 'model_cv_8', 'model_cv_9']

    num_models = len(model_dirs)
    cv_preds = []

    for model_file in model_dirs:
        imcnn = SegImcnn(adapt_data=PartNetGraspDataset(PARTNET_GRASP, set_type=3, only_signal=True))
        imcnn.load_state_dict(torch.load(models_path + model_file + "/model.zip"))
        preds = imcnn([signal, bc]).detach().cpu().numpy()  # (num_points, num_classes)
        cv_preds.append(np.argmax(preds, axis=1))

    cv_preds = np.stack(cv_preds, axis=0)  # Shape: (num_models, num_points)
    num_points = cv_preds.shape[1]
    final_preds = np.zeros(num_points, dtype=int)
    misclassified_idxs = []

    for i in range(num_points):
        point_preds = cv_preds[:, i]  # Shape: (num_models,)
        gt_label = labels[i]
        pred_counts = Counter(point_preds)
        most_common_label, count = pred_counts.most_common(1)[0]

        if voting == "majority":
            final_preds[i] = most_common_label
            # majority of models were wrong
            if most_common_label != gt_label:
                misclassified_idxs.append(i)

        elif voting == "consensus":
            # all models must be wrong
            if np.all(point_preds != gt_label):
                final_preds[i] = most_common_label  # still give the majority prediction
                misclassified_idxs.append(i)
            else:
                final_preds[i] = gt_label  # assume correct if not full consensus

        elif voting == "percentage":
            # compute fraction of models that predicted incorrectly
            num_wrong = np.sum(point_preds != gt_label)
            if (num_wrong / num_models) >= percentage_misclass:
                final_preds[i] = most_common_label
                misclassified_idxs.append(i)
            else:
                final_preds[i] = gt_label

        else:
            raise ValueError(f"Unknown voting strategy: {voting}")

    return final_preds, misclassified_idxs


import random


def randomly_ranked_mesh_changes(mesh_changes_dict, seed=None):
    """
    Randomly flattens and ranks the mesh changes.

    Args:
        mesh_changes_dict: dict where key=mesh_idx, value=list of point indices to change
        seed: optional random seed for reproducibility

    Returns:
        List of tuples: (mesh_idx, point_idx, rank)
    """
    if seed is not None:
        random.seed(seed)

    all_changes = []
    for mesh_idx, point_list in mesh_changes_dict.items():
        for point_idx in point_list:
            all_changes.append((mesh_idx, point_idx))

    random.shuffle(all_changes)

    ranked_changes = [(mesh_idx, point_idx, rank) for rank, (mesh_idx, point_idx) in enumerate(all_changes)]

    return ranked_changes


# LOGGING_DIR = f"{EXPERIMENT_DIRECTORY}/logs/cv/"
PARTNET_GRASP = "/home/iroberts/projects/VisMeshSegmentation/improve_mesh_segmentation/datasets/partnet_grasp.zip"
if __name__ == "__main__":
    # models_path = "/home/iroberts/projects/VisMeshSegmentation/run_through/logs/behanvior_exp/cv/"
    # model_dirs = ['model_cv_0', 'model_cv_1', 'model_cv_2', 'model_cv_3', 'model_cv_4', 'model_cv_5', 'model_cv_6','model_cv_7', 'model_cv_8', 'model_cv_9']
    # X = np.arange(100)
    # cv = KFold(n_splits=10)
    # cv_pred_dict = []
    # for i, ((train_idxs,test_idxs), model_file) in enumerate(zip(cv.split(X),model_dirs)):
    #     imcnn = SegImcnn(adapt_data=PartNetGraspDataset(PARTNET_GRASP, set_type=3, only_signal=True))
    #     imcnn.load_state_dict(torch.load(models_path+model_file+ "/model.zip"))
    #     test_dataset = list(processed_partnet_grasp_generator(PARTNET_GRASP, set_type=3,set_indices=test_idxs ))
    #     for mesh_idx,((signal, bc), og_labels) in zip(test_idxs,test_dataset):
    #         preds = imcnn([signal,bc]).detach().cpu().numpy().tolist()
    #         cv_pred_dict.append({
    #             "mesh_idx": mesh_idx.item(),
    #             "preds": preds,
    #         })
    #
    # with open("/home/iroberts/projects/VisMeshSegmentation/run_through/logs/behanvior_exp/cv/cv_out_of_sample_preds.json", "w") as f:
    #     json.dump(cv_pred_dict, f)
    path_to_corrections = "/home/iroberts/projects/VisMeshSegmentation/run_through/corrections/behavior_exp/"
    #
    dataset = list(processed_partnet_grasp_generator(PARTNET_GRASP, set_type=3))
    cv_preds = []
    mesh_preds_dict = {}
    mesh_changes_dict = {}
    for mesh_idx, ((signal,bc),labels) in enumerate(dataset):
        mesh_preds, points_to_change = cv_model_pred(signal,bc,labels,voting="majority")
        mesh_preds_dict[mesh_idx] = mesh_preds
        mesh_changes_dict[mesh_idx] = points_to_change

    ranked_list = randomly_ranked_mesh_changes(mesh_changes_dict, seed=42)
    os.mkdir(path_to_corrections + "ensemble_majority_baseline")
    file_name = "ensemble_majority_baseline" + "/ensemble_majority_baseline" + ".csv"
    points_to_change = ranked_list[:9092]
    write_label_changes(path_to_corrections + file_name, points_to_change, mesh_preds_dict)

    # for i in list(range(500, 9500, 500)):
    #     if i > len(ranked_list):
    #         file_name = "cv_majority_baseline" + "/cv_majority_baseline_"  + str(i) + ".csv"
    #         points_to_change = ranked_list[:len(ranked_list)]
    #         write_label_changes(path_to_corrections + file_name, points_to_change, mesh_preds_dict)
    #     else:
    #
    #         file_name = "cv_majority_baseline" + "/cv_majority_baseline_"  + str(i) + ".csv"
    #         points_to_change = ranked_list[:i]
    #         write_label_changes(path_to_corrections + file_name, points_to_change, mesh_preds_dict)
    # write_label_changes(csv_file_path, ranked_list, mesh_preds)








    # import json
    #
    # with open("../cv_out_of_sample_preds.json", "w") as f:
    #     json.dump(cv_pred_dict, f)
