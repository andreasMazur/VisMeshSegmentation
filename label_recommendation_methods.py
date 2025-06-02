import collections
import json
from collections import defaultdict
import random

import numpy as np
import scipy as sp
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from torch import nn
from improve_mesh_segmentation.data_correction.correct_sub_partnet import pred_wrapper

from filter_methods import misclassifications_uncertainty_baseline, misclassifications_influence_baseline, \
    influence_baseline, deepview_variants, influence_uncertainty_combination_baseline, deepview_kmeans
from improve_mesh_segmentation.partnet_grasp.dataset import PartNetGraspDataset, processed_partnet_grasp_generator
from improve_mesh_segmentation.data_correction.correct_sub_partnet import embed
from improve_mesh_segmentation.training.imcnn import SegImcnn

from helper_functions import *

og_data_path = "/home/iroberts/projects/VisMeshSegmentation/improve_mesh_segmentation/datasets/partnet_grasp.zip"

model_path = "/home/iroberts/projects/VisMeshSegmentation/run_through/logs/model.zip"


def global_mesh_sort(neighbors_dict, values_dict, descending=False):
    combined = []
    for mesh_idx in neighbors_dict:
        vertex_indices = neighbors_dict[mesh_idx]
        values = values_dict[mesh_idx]
        for v_idx, val in zip(vertex_indices, values):
            combined.append((mesh_idx, v_idx, val))


    # Sort globally by value
    combined_sorted = sorted(combined, key=lambda x: x[2],reverse=descending)

    # # Return top-k if specified
    # if k is not None:
    #     return combined_sorted[:k]
    return combined_sorted,combined




if __name__ == "__main__":
    # Load shared datasets into memory (as lists)
    og_dataset = list(processed_partnet_grasp_generator(og_data_path, set_type=0))
    # corrected_dataset = list(processed_partnet_grasp_generator(corrected_data_path, set_type=0))

    imcnn = SegImcnn(adapt_data=PartNetGraspDataset(og_data_path, set_type=0, only_signal=True))
    imcnn.load_state_dict(torch.load(model_path))
    classification_head = imcnn.model.output_dense

    dropout_prob = 0.5
    stochastic_model = StochasticModel(classification_head, dropout_prob)


    def pred_wrapper(data):
        """Get the predicted probabilities of an IMCNN.

        Parameters
        ----------
        data: torch.Tensor
            The data which shall be embedded.
        model: SegImcnn
            The model that embeds the data.
        """
        return sp.special.softmax(imcnn.model.output_dense(torch.tensor(data).float()).detach().numpy(), axis=-1)



    mesh_indices = {}
    mesh_value1 = {}
    mesh_value2 = {}
    mesh_preds = {}
    for mesh_idx, ((signal, bc), labels) in enumerate(og_dataset):
        print(".....Correcting Mesh Index: ", mesh_idx)
        labels = np.array(labels)
        embeddings = embed(imcnn, [signal, bc])
        embeddings = torch.tensor(embeddings)
        preds = np.argmax(classification_head(embeddings).detach().numpy(), axis=1)
        # idxs,unc, inf  = deepview_variants(pred_wrapper, embeddings, labels, stochastic_model,mesh_idx)
        idxs, unc = deepview_kmeans(pred_wrapper, embeddings, labels, preds, stochastic_model)
        # idxs,unc = misclassifications_uncertainty_baseline(embeddings,stochastic_model,labels,preds)
        # idxs,inf = misclassifications_influence_baseline(embeddings,labels,preds,mesh_idx)
        # idxs,inf = influence_baseline(embeddings, mesh_idx)
        # idxs, inf =  influence_uncertainty_combination_baseline(embeddings, mesh_idx, stochastic_model, labels)
        mesh_indices[mesh_idx] = idxs
        mesh_value1[mesh_idx] = unc
        # mesh_value2[mesh_idx] = inf
        mesh_preds[mesh_idx] = preds

    path_to_corrections = "/home/iroberts/projects/VisMeshSegmentation/run_through/corrections/"
    sorted_by_unc, not_sorted_by_unc = global_mesh_sort(mesh_indices, mesh_value1)
    # sorted_by_inf, not_sorted_by_inf = global_mesh_sort(mesh_indices, mesh_value2,descending=True)
    for k in list(range(500,9500,500)):
        if k > len(sorted_by_unc):
            # file_name = "deepview_background_sorted_unc/deepview_background_sorted_unc_k_" + str(k) + ".csv"
            # points_to_change = sorted_by_unc[:len(sorted_by_unc)]
            # write_label_changes(path_to_corrections+file_name, points_to_change, mesh_preds)

            file_name = "deepview_kmeans/deepview_kmeans_k_" + str(k) + ".csv"
            points_to_change = sorted_by_unc[:len(sorted_by_unc)]
            write_label_changes(path_to_corrections + file_name, points_to_change, mesh_preds)

            # file_name = "misclassification_sorted_inf/misclassification_sorted_inf_k_" + str(k) + ".csv"
            # points_to_change = sorted_by_inf[:len(sorted_by_inf)]
            # write_label_changes(path_to_corrections + file_name, points_to_change, mesh_preds)

            # file_name = "deepview_background_sorted_inf/deepview_background_sorted_inf_k_" + str(k) + ".csv"
            # points_to_change = sorted_by_inf[:len(sorted_by_inf)]
            # write_label_changes(path_to_corrections + file_name, points_to_change, mesh_preds)

            # file_name = "influence_baseline/influence_baseline_k_" + str(k) + ".csv"
            # points_to_change = sorted_by_inf[:len(sorted_by_inf)]
            # write_label_changes(path_to_corrections + file_name, points_to_change, mesh_preds)

            # file_name = "influence_uncertainty_combination_baseline/influence_uncertainty_combination_baseline_k_" + str(k) + ".csv"
            # points_to_change = sorted_by_inf[:len(sorted_by_inf)]
            # write_label_changes(path_to_corrections + file_name, points_to_change, mesh_preds)

            # random_file_name = "deepview_background_random/deepview_background_random_k_" + str(k) + ".csv"
            # random.shuffle(not_sorted_by_unc)
            # points_to_change = not_sorted_by_unc[:len(sorted_by_unc)]
            # write_label_changes(path_to_corrections + random_file_name, points_to_change, mesh_preds)

        else:
            # file_name = "deepview_background_sorted_unc/deepview_background_sorted_unc_k_" + str(k) + ".csv"
            # points_to_change = sorted_by_unc[:k]
            # write_label_changes(path_to_corrections + file_name, points_to_change, mesh_preds)

            file_name = "deepview_kmeans/deepview_kmeans_k_" + str(k) + ".csv"
            points_to_change = sorted_by_unc[:k]
            write_label_changes(path_to_corrections + file_name, points_to_change, mesh_preds)

            # file_name = "misclassification_sorted_inf/misclassification_sorted_inf_k_" + str(k) + ".csv"
            # points_to_change = sorted_by_inf[:k]
            # write_label_changes(path_to_corrections + file_name, points_to_change, mesh_preds)

            # file_name = "influence_baseline/influence_baseline_k_" + str(k) + ".csv"
            # points_to_change = sorted_by_inf[:k]
            # write_label_changes(path_to_corrections + file_name, points_to_change, mesh_preds)

            # file_name = "influence_uncertainty_combination_baseline/influence_uncertainty_combination_baseline_k_"+ str(k) + ".csv"
            # points_to_change = sorted_by_inf[:k]
            # write_label_changes(path_to_corrections + file_name, points_to_change, mesh_preds)

            # file_name = "deepview_background_sorted_inf/deepview_background_sorted_inf_k_" + str(k) + ".csv"
            # points_to_change = sorted_by_inf[:k]
            # write_label_changes(path_to_corrections + file_name, points_to_change, mesh_preds)

            # random_file_name = "deepview_background_random/deepview_background_random_k_" + str(k) + ".csv"
            # random.shuffle(not_sorted_by_unc)
            # points_to_change = not_sorted_by_unc[:k]
            # write_label_changes(path_to_corrections + random_file_name, points_to_change, mesh_preds)



