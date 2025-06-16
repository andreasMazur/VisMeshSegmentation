import os

import scipy as sp
import torch

from relabel_helpers.comparison_methods.filter_methods import deepview_kmeans, deepview_kmeans_bg, deepview_dbscan, \
    deepview_knn, deepview_sup_kmeans, _kmeans, _knn
from improve_mesh_segmentation.partnet_grasp.dataset import PartNetGraspDataset, processed_partnet_grasp_generator
from improve_mesh_segmentation.data_correction.correct_sub_partnet import embed
from improve_mesh_segmentation.training.imcnn import SegImcnn

from relabel_helpers.helper_functions import *

og_data_path = "/home/iroberts/projects/VisMeshSegmentation/improve_mesh_segmentation/datasets/partnet_grasp.zip"

model_path = "/home/iroberts/projects/VisMeshSegmentation/run_through/logs/model.zip"
corrected_data_path = "/home/iroberts/projects/VisMeshSegmentation/improve_mesh_segmentation/datasets/partnet_grasp_corrected.zip"


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
    corrected_dataset = list(processed_partnet_grasp_generator(corrected_data_path, set_type=0))

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
    mesh_labels = {}
    for mesh_idx, (((signal, bc), labels),((_, _), cor_labels)) in enumerate(zip(og_dataset,corrected_dataset)):
        print(".....Correcting Mesh Index: ", mesh_idx)
        labels = np.array(labels)
        embeddings = embed(imcnn, [signal, bc])
        embeddings = torch.tensor(embeddings)
        preds = np.argmax(classification_head(embeddings).detach().numpy(), axis=1)
        # idxs,unc, new_labels  = deepview_knn(pred_wrapper, embeddings, labels, stochastic_model,mesh_idx)
        idxs, unc, new_labels = _knn(embeddings, labels,stochastic_model)

        # idxs, unc, new_labels = deepview_kmeans(pred_wrapper, embeddings, labels, preds, stochastic_model)
        # idxs, unc, new_labels = deepview_dbscan(pred_wrapper, embeddings, labels, preds, stochastic_model)
        # idxs, unc = _lvq(embeddings, labels,preds, stochastic_model)
        # print("Rand with bad labels:" + str(rand_score(labels,idxs)) + " \nADJ Rand with bad labels:" + str(adjusted_rand_score(labels,idxs)))
        # print("Rand with good labels:" + str(rand_score(cor_labels, idxs)) + " \nADJ Rand with good labels :" + str(
        #     adjusted_rand_score(cor_labels, idxs)))
        # idxs,unc = misclassifications_uncertainty_baseline(embeddings,stochastic_model,labels,preds)
        # idxs,inf = misclassifications_influence_baseline(embeddings,labels,preds,mesh_idx)
        # idxs,inf = influence_baseline(embeddings, mesh_idx)
        # idxs, inf =  influence_uncertainty_combination_baseline(embeddings, mesh_idx, stochastic_model, labels)
        mesh_indices[mesh_idx] = idxs
        mesh_value1[mesh_idx] = unc
        # mesh_value2[mesh_idx] = inf
        mesh_labels[mesh_idx] = new_labels

    path_to_corrections = "/home/iroberts/projects/VisMeshSegmentation/run_through/corrections/"

    os.mkdir(path_to_corrections + "iterative_knn")
    sorted_by_unc, not_sorted_by_unc = global_mesh_sort(mesh_indices, mesh_value1)
    # sorted_by_inf, not_sorted_by_inf = global_mesh_sort(mesh_indices, mesh_value2,descending=True)
    for k in list(range(500,9500,500)):
        if k > len(sorted_by_unc):
            # file_name = "deepview_background_sorted_unc/deepview_background_sorted_unc_k_" + str(k) + ".csv"
            # points_to_change = sorted_by_unc[:len(sorted_by_unc)]
            # write_label_changes(path_to_corrections+file_name, points_to_change, mesh_preds)

            # file_name = "deepview_kmeans7/deepview_kmeans_7_" + str(k) + ".csv"
            # points_to_change = sorted_by_unc[:len(sorted_by_unc)]
            # write_label_changes(path_to_corrections + file_name, points_to_change, mesh_preds)

            file_name = "iterative_knn/iterative_knn_" + str(k) + ".csv"
            points_to_change = sorted_by_unc[:len(sorted_by_unc)]
            write_label_changes(path_to_corrections + file_name, points_to_change, mesh_labels)

            # file_name = "lvq/lvq_k_" + str(k) + ".csv"
            # points_to_change = sorted_by_unc[:len(sorted_by_unc)]
            # write_label_changes(path_to_corrections + file_name, points_to_change, mesh_preds)

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

            # file_name = "lvq/lvq_k_" + str(k) + ".csv"
            # points_to_change = sorted_by_unc[:k]
            # write_label_changes(path_to_corrections + file_name, points_to_change, mesh_preds)

            file_name = "iterative_knn/iterative_knn_" + str(k) + ".csv"
            points_to_change = sorted_by_unc[:k]
            write_label_changes(path_to_corrections + file_name, points_to_change, mesh_labels)

            # file_name = "deepview_kmeans7/deepview_kmeans_7_" + str(k) + ".csv"
            # points_to_change = sorted_by_unc[:k]
            # write_label_changes(path_to_corrections + file_name, points_to_change, mesh_preds)

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



