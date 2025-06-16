import json
import os

from relabel_helpers.comparison_methods.filter_methods import knn_label_correction
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


def get_embeddings_labels_preds_with_idx_map(og_dataset, imcnn, classification_head, mesh_range):
    """
    Returns embeddings, labels, predictions, and a mesh-to-global-index mapping.

    Parameters:
        og_dataset: List or dataset of ((signal, bc), labels)
        imcnn: Feature extractor model
        classification_head: Classification head model (on embeddings)
        mesh_range: Iterable of mesh indices to process (e.g., range(0, 5))

    Returns:
        all_embeddings: torch.Tensor of shape (N, D)
        all_labels: numpy array of shape (N,)
        all_preds: numpy array of shape (N,)
        mesh_to_global_idx: dict mapping mesh_idx -> list of global vertex indices
    """
    all_embeddings = []
    all_labels = []
    all_preds = []
    pred_dict = {}
    unc_dict = {}
    mesh_to_global_idx = {}

    global_vertex_idx = 0
    dropout_prob = 0.5
    stochastic_model = StochasticModel(classification_head, dropout_prob)

    for mesh_idx in mesh_range:
        (signal, bc), labels = og_dataset[mesh_idx]
        print(".....Processing Mesh Index:", mesh_idx)

        labels = np.array(labels)
        embeddings = embed(imcnn, [signal, bc])  # shape (V, D) or list of vectors
        embeddings = torch.tensor(embeddings)

        with torch.no_grad():
            preds = classification_head(embeddings)
            preds = preds.detach().numpy()
            preds = np.argmax(preds, axis=1)

        dataset = EmbeddingDataset(embeddings, labels)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)

        # Get output Distribution
        predictions = predict_with_uncertainty_batched(stochastic_model, dataloader, n_iter=30)
        predictions = predictions.numpy()

        # Calculate entropy
        a, prob_mat = uncertainty_matrices(predictions)
        t, e, a = entropy_uncertainty(prob_mat)

        num_vertices = embeddings.shape[0]
        current_indices = list(range(global_vertex_idx, global_vertex_idx + num_vertices))
        mesh_to_global_idx[mesh_idx] = current_indices
        global_vertex_idx += num_vertices

        all_embeddings.append(embeddings)
        all_labels.append(labels)
        all_preds.append(preds)
        pred_dict[mesh_idx] = preds
        unc_dict[mesh_idx] = t

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    return all_embeddings, all_labels, all_preds, mesh_to_global_idx,pred_dict,unc_dict


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
        models_path = "/home/iroberts/projects/VisMeshSegmentation/run_through/logs/cv/"
        model_dirs = ['model_cv_0', 'model_cv_1', 'model_cv_2', 'model_cv_3', 'model_cv_4', 'model_cv_5', 'model_cv_6']
        num_models = len(model_dirs)
        cv_preds = []

        for model_file in model_dirs:
            imcnn = SegImcnn(adapt_data=PartNetGraspDataset(og_data_path, set_type=0, only_signal=True))
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
            pred_counts = collections.Counter(point_preds)
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

    imcnn = SegImcnn(adapt_data=PartNetGraspDataset(og_data_path, set_type=0, only_signal=True))
    imcnn.load_state_dict(torch.load(model_path))
    classification_head = imcnn.model.output_dense

    with open('cv_out_of_sample_preds.json', 'r') as f:
        cv_preds = json.load(f)

    # Convert to a dictionary for quick mesh_idx lookup
    cv_pred_dict = {entry['mesh_idx']: entry['preds'] for entry in cv_preds}

    # for i in range(0,70,20):
    embs, labels, preds, map,pred_dict, g_unc_dict = get_embeddings_labels_preds_with_idx_map(og_dataset, imcnn, classification_head,
                                                                            range(0, 70))
    all_preds = []
    for key in cv_pred_dict.keys():
        all_preds.append(np.argmax(cv_pred_dict[key],axis=1))
    cv_all_preds = np.concatenate(all_preds, axis=0)
    # ks = list(range(1,11))
    # for k in ks:
    #     idxs,unc_dict,new_labels = lvq_label_correction(embs, labels, preds, map, g_unc_dict,k)
    #
    #     # idxs, unc_dict, new_labels = dbscan_label_correction(embs, labels, preds, map, unc_dict)
    #     sorted_by_unc, not_sorted_by_unc = global_mesh_sort(idxs,unc_dict)

        # path_to_corrections = "/home/iroberts/projects/VisMeshSegmentation/run_through/corrections/"
        # os.mkdir(path_to_corrections + "global_lvq_" + str(k))
        # for i in list(range(500,9500,500)):
        #     if i > len(sorted_by_unc):
        #         file_name = "global_lvq_" + str(k)+ "/global_lvq_"+ str(k)+ "_" + str(i) + ".csv"
        #         points_to_change = sorted_by_unc[:len(sorted_by_unc)]
        #         write_label_changes(path_to_corrections + file_name, points_to_change, new_labels)
        #
        #     else:
        #         file_name = "global_lvq_" + str(k)+ "/global_lvq_"+ str(k)+ "_" + str(i) + ".csv"
        #         points_to_change = sorted_by_unc[:i]
        #         write_label_changes(path_to_corrections + file_name, points_to_change, new_labels)

    ks = [5,10,25,50,100,250,500,1000]
    for k in ks:
        idxs,unc_dict,new_labels = knn_label_correction(embs, labels, cv_all_preds, map, g_unc_dict,k)

        # idxs, unc_dict, new_labels = dbscan_label_correction(embs, labels, preds, map, unc_dict)
        sorted_by_unc, not_sorted_by_unc = global_mesh_sort(idxs,unc_dict)

        path_to_corrections = "/home/iroberts/projects/VisMeshSegmentation/run_through/corrections/"
        os.mkdir(path_to_corrections + "global_cvknn_" + str(k))
        for i in list(range(500,9500,500)):
            if i > len(sorted_by_unc):
                file_name = "global_cvknn_" + str(k)+ "/global_cvknn_"+ str(k)+ "_" + str(i) + ".csv"
                points_to_change = sorted_by_unc[:len(sorted_by_unc)]
                write_label_changes(path_to_corrections + file_name, points_to_change, new_labels)
            else:

                file_name = "global_cvknn_" + str(k)+ "/global_cvknn_"+ str(k)+ "_" + str(i) + ".csv"
                points_to_change = sorted_by_unc[:i]
                write_label_changes(path_to_corrections + file_name, points_to_change, new_labels)




