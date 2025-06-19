import os

import scipy as sp
import torch

from dataset_creator_global_methods import global_mesh_sort, get_embeddings_labels_preds_with_idx_map
from relabel_helpers.comparison_methods.filter_methods import deepview_kmeans, deepview_kmeans_bg, \
    deepview_knn, _kmeans, _knn, knn_label_correction, lvq_label_correction, \
    kmeans_label_correction, supervised_kmeans_label_correction, \
    misclassifications_uncertainty_baseline
from improve_mesh_segmentation.partnet_grasp.dataset import PartNetGraspDataset, processed_partnet_grasp_generator
from improve_mesh_segmentation.data_correction.correct_sub_partnet import embed
from improve_mesh_segmentation.training.imcnn import SegImcnn

from relabel_helpers.helper_functions import *

og_data_path = "/home/iroberts/projects/VisMeshSegmentation/improve_mesh_segmentation/datasets/partnet_grasp.zip"

model_path = "/home/iroberts/projects/VisMeshSegmentation/run_through/logs/model.zip"
corrected_data_path = "/home/iroberts/projects/VisMeshSegmentation/improve_mesh_segmentation/datasets/partnet_grasp_corrected.zip"


import os
import json
import numpy as np
import torch
import scipy as sp
from tqdm import tqdm

def run_label_correction(
    method_name: str,
    imcnn,
    classification_head,
    dataset,
    corrected_dataset=None,
    mesh_range=range(0, 100),
    path_to_corrections="/home/iroberts/projects/VisMeshSegmentation/run_through/corrections/behavior_exp/",
    method_parameter=None,
embs=None,
        labels=None,
        preds=None,
        map=None, pred_dict=None, g_unc_dict=None
):
    dropout_prob = 0.5
    stochastic_model = StochasticModel(classification_head, dropout_prob)

    def pred_wrapper(data):
        return sp.special.softmax(imcnn.model.output_dense(torch.tensor(data).float()).detach().numpy(), axis=-1)

    is_global = method_name.startswith("global_")

    if is_global:
        # assert mesh_range is not None, "You must specify mesh_range for global methods."



        # Dispatch method
        if "misclass_unc" in method_name:
            idxs, unc_dict, new_labels = misclassifications_uncertainty_baseline(labels,preds,map,g_unc_dict)
        elif "knn_observed" in method_name:
            idxs, unc_dict, new_labels = knn_label_correction(embs, labels, labels, map, g_unc_dict, method_parameter)
        elif "knn_pred" in method_name:
            idxs, unc_dict, new_labels = knn_label_correction(embs, labels, preds, map, g_unc_dict, method_parameter)
        elif "lvq_observed" in method_name:
            idxs, unc_dict, new_labels = lvq_label_correction(embs, labels,map, g_unc_dict, method_parameter)
        elif "lvq_pred" in method_name:
            idxs, unc_dict, new_labels = lvq_label_correction(embs, preds, map, g_unc_dict, method_parameter)
        elif "kmeans_observed" in method_name and "supervised" not in method_name:
            idxs, unc_dict, new_labels = kmeans_label_correction(embs, labels, labels, map, g_unc_dict,method_parameter)
        elif "kmeans_pred" in method_name and "supervised" not in method_name:
            idxs, unc_dict, new_labels = kmeans_label_correction(embs, labels, preds, map, g_unc_dict,method_parameter)
        elif "supervised_kmeans_pred" in method_name:
            idxs, unc_dict, new_labels = supervised_kmeans_label_correction(embs,  preds,map, g_unc_dict, method_parameter)
        elif "supervised_kmeans_observed" in method_name:
            idxs, unc_dict, new_labels = supervised_kmeans_label_correction(embs, labels,map, g_unc_dict, method_parameter)
        else:
            raise NotImplementedError(f"Unknown method: {method_name}")

        sorted_by_unc, _ = global_mesh_sort(idxs, unc_dict)
        method_dir = f"{method_name}"
        os.makedirs(path_to_corrections + method_dir, exist_ok=True)

            # for i in ks:
            #     limit = min(i, len(sorted_by_unc))
        points_to_change = sorted_by_unc[:9092]
        file_path = f"{method_dir}/{method_dir}_{method_parameter}.csv"
        write_label_changes(os.path.join(path_to_corrections, file_path), points_to_change, new_labels)

    else:
        mesh_indices = {}
        mesh_value1 = {}
        mesh_labels = {}
        for mesh_idx, (((signal, bc), labels), ((_, _), cor_labels)) in tqdm(enumerate(zip(dataset, corrected_dataset))):
            labels = np.array(labels)
            embeddings = embed(imcnn, [signal, bc])
            embeddings = torch.tensor(embeddings)
            preds = np.argmax(classification_head(embeddings).detach().numpy(),axis=1)

            if method_name == "deepview_knn_observed":
                idxs, unc, new_labels = deepview_knn(pred_wrapper, embeddings, labels, labels, stochastic_model,method_parameter)
            elif method_name == "deepview_kmeans_bg_observed":
                idxs, unc, new_labels = deepview_kmeans_bg(pred_wrapper, embeddings, labels, labels, stochastic_model,method_parameter)
            elif method_name == "deepview_knn_pred":
                idxs, unc, new_labels = deepview_knn(pred_wrapper, embeddings, labels, preds, stochastic_model,method_parameter)
            elif method_name == "deepview_kmeans_bg_pred":
                idxs, unc, new_labels = deepview_kmeans_bg(pred_wrapper, embeddings, labels, preds, stochastic_model,method_parameter)
            elif method_name == "deepview_kmeans_pred":
                idxs, unc, new_labels = deepview_kmeans(pred_wrapper, embeddings, labels, preds, stochastic_model,method_parameter)
            elif method_name == "deepview_kmeans_observed":
                idxs, unc, new_labels = deepview_kmeans(pred_wrapper, embeddings, labels, labels, stochastic_model,method_parameter)
            elif method_name == "iterative_knn_observed":
                idxs, unc, new_labels = _knn(embeddings, labels, labels, stochastic_model,method_parameter)
            elif method_name == "iterative_kmeans_observed":
                idxs, unc, new_labels = _kmeans(embeddings, labels, labels, stochastic_model,method_parameter)
            elif method_name == "iterative_knn_pred":
                idxs, unc, new_labels = _knn(embeddings, labels, preds, stochastic_model,method_parameter)
            elif method_name == "iterative_kmeans_pred":
                idxs, unc, new_labels = _kmeans(embeddings, labels, preds, stochastic_model,method_parameter)
            else:
                raise NotImplementedError(f"Unknown method: {method_name}")

            mesh_indices[mesh_idx] = idxs
            mesh_value1[mesh_idx] = unc
            mesh_labels[mesh_idx] = new_labels

        sorted_by_unc, _ = global_mesh_sort(mesh_indices, mesh_value1)
        method_dir = f"{method_name}"
        os.makedirs(path_to_corrections + method_dir, exist_ok=True)

        # for k in ks:
        #     limit = min(k, len(sorted_by_unc))
        points_to_change = sorted_by_unc[:9092]
        file_path = f"{method_dir}/{method_name}_{method_parameter}.csv"
        write_label_changes(os.path.join(path_to_corrections, file_path), points_to_change, mesh_labels)


if __name__ == "__main__":
    og_dataset = list(processed_partnet_grasp_generator(og_data_path, set_type=3))
    corrected_dataset = list(processed_partnet_grasp_generator(corrected_data_path, set_type=3))
    imcnn = SegImcnn(adapt_data=PartNetGraspDataset(og_data_path, set_type=3, only_signal=True))
    imcnn.load_state_dict(torch.load(model_path))
    classification_head = imcnn.model.output_dense

    embs, labels, preds, map, pred_dict, g_unc_dict = get_embeddings_labels_preds_with_idx_map(
        og_dataset, imcnn, classification_head, range(1,100)
    )

    # run_label_correction("global_misclass_unc", imcnn, classification_head, og_dataset, corrected_dataset,
    #                      embs=embs, labels=labels, preds=preds, map=map, pred_dict=pred_dict, g_unc_dict=g_unc_dict)
    # print("*********************************done with global misclass*************************************************")
    #
    # for i in  [5,10,25,50,100,250,500,1000,2000,3000,4000,5000,]:
    #     run_label_correction("global_knn_observed", imcnn, classification_head, og_dataset, corrected_dataset,
    #                      method_parameter=i, embs=embs, labels=labels, preds=preds, map=map, pred_dict=pred_dict, g_unc_dict=g_unc_dict)
    #
    # print("*********************************done with global_knn_observed*************************************************")
    #
    # for i in  [5,10,25,50,100,250,500,1000,2000,3000,4000,5000,]:
    #     run_label_correction("global_knn_pred", imcnn, classification_head, og_dataset, corrected_dataset,
    #                      method_parameter=i, embs=embs, labels=labels, preds=preds, map=map, pred_dict=pred_dict, g_unc_dict=g_unc_dict)
    # print(
    #     "*********************************done with global_knn_pred*************************************************")
    # for i in range(1,6):
    #     run_label_correction("global_lvq_observed", imcnn, classification_head, og_dataset, corrected_dataset,method_parameter=i,
    #                           embs=embs, labels=labels, preds=preds, map=map, pred_dict=pred_dict, g_unc_dict=g_unc_dict)
    #
    # print(
    #     "*********************************done with global_lvq_observed*************************************************")
    #
    #
    # for i in range(1, 6):
    #     run_label_correction("global_lvq_pred", imcnn, classification_head, og_dataset, corrected_dataset,
    #                              method_parameter=i, embs=embs, labels=labels, preds=preds, map=map, pred_dict=pred_dict, g_unc_dict=g_unc_dict)

    print(
        "*********************************done with global_lvq_pred*************************************************")
    for i in range(2,11):
        run_label_correction("global_kmeans_observed", imcnn, classification_head, og_dataset, corrected_dataset,method_parameter=i,
                             embs=embs, labels=labels, preds=preds, map=map, pred_dict=pred_dict, g_unc_dict=g_unc_dict)

    print(
        "*********************************done with global_kmeans_observed*************************************************")
    for i in range(2, 11):
        run_label_correction("global_kmeans_pred", imcnn, classification_head, og_dataset, corrected_dataset,
                                 method_parameter=i, embs=embs, labels=labels, preds=preds, map=map, pred_dict=pred_dict, g_unc_dict=g_unc_dict)

    print(
        "*********************************done with global_kmeans_pred*************************************************")

    for i in range(1,6):
        run_label_correction("global_supervised_kmeans_observed", imcnn, classification_head, og_dataset, corrected_dataset,method_parameter=i,
                              embs=embs, labels=labels, preds=preds, map=map, pred_dict=pred_dict, g_unc_dict=g_unc_dict)

    print(
        "*********************************done with global_supervised_kmeans_observed*************************************************")

    for i in range(1, 6):
        run_label_correction("global_supervised_kmeans_pred", imcnn, classification_head, og_dataset, corrected_dataset,
                                 method_parameter=i, embs=embs, labels=labels, preds=preds, map=map, pred_dict=pred_dict, g_unc_dict=g_unc_dict)

    print(
        "*********************************done with global_supervised_kmeans_pred*************************************************")

    for i in [5,10,15,25,50]:
        run_label_correction("deepview_knn_observed", imcnn, classification_head, og_dataset,
                                 corrected_dataset,
                                 method_parameter=i)
    print(
            "*********************************done with deepview_knn_observed*************************************************")

    for i in range(2, 6):
        run_label_correction("deepview_kmeans_bg_observed", imcnn, classification_head, og_dataset,
                                 corrected_dataset,
                                 method_parameter=i)
    print(
            "*********************************done with deepview_kmeans_bg_observed*************************************************")


    for i in [5,10,15,25,50]:
        run_label_correction("deepview_knn_pred", imcnn, classification_head, og_dataset,
                                 corrected_dataset,
                                 method_parameter=i)

    print(
            "*********************************done with deepview_knn_pred*************************************************")

    for i in range(2, 6):
        run_label_correction("deepview_kmeans_bg_pred", imcnn, classification_head, og_dataset,
                                 corrected_dataset,
                                 method_parameter=i)

    print(
            "*********************************done with deepview_kmeans_bg_pred*************************************************")
    for i in range(2, 6):
        run_label_correction("deepview_kmeans_pred", imcnn, classification_head, og_dataset,
                                 corrected_dataset,
                                 method_parameter=i)
    print(
        "*********************************done with deepview_kmeans_pred*************************************************")
    for i in range(2, 6):
        run_label_correction("deepview_kmeans_observed", imcnn, classification_head, og_dataset,
                                 corrected_dataset,
                                 method_parameter=i)
    print(
        "*********************************done with deepview_kmeans_observed*************************************************")

    for i in [5,10,15,25,50]:
        run_label_correction("iterative_knn_observed", imcnn, classification_head, og_dataset,
                                 corrected_dataset,
                                 method_parameter=i)

    print(
        "*********************************done with iterative_knn_observed*************************************************")

    for i in range(2, 6):
        run_label_correction("iterative_kmeans_observed", imcnn, classification_head, og_dataset,
                                 corrected_dataset,
                                 method_parameter=i)

    print(
        "*********************************done with iterative_kmeans_observed*************************************************")

    for i in [5,10,15,25,50]:
        run_label_correction("iterative_knn_pred", imcnn, classification_head, og_dataset,
                                 corrected_dataset,
                                 method_parameter=i)

    print(
        "*********************************done with iterative_knn_pred*************************************************")
    for i in range(2, 6):
        run_label_correction("iterative_kmeans_pred", imcnn, classification_head, og_dataset,
                                 corrected_dataset,
                                 method_parameter=i)

print(
        "*********************************done with iterative_kmeans_pred*************************************************")

