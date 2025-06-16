import json
import os

from cleanlab.filter import find_label_issues

from improve_mesh_segmentation.partnet_grasp.dataset import PartNetGraspDataset, processed_partnet_grasp_generator
from improve_mesh_segmentation.data_correction.correct_sub_partnet import embed
from improve_mesh_segmentation.training.imcnn import SegImcnn

from relabel_helpers.helper_functions import *

og_data_path = "/home/iroberts/projects/VisMeshSegmentation/improve_mesh_segmentation/datasets/partnet_grasp.zip"

model_path = "/home/iroberts/projects/VisMeshSegmentation/run_through/logs/model.zip"
# corrected_data_path = "/improve_mesh_segmentation/datasets/partnet_grasp_corrected.zip"


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

def sigmoid(z):
    return 1/(1 + np.exp(-z))

if __name__ == "__main__":
    og_dataset = list(processed_partnet_grasp_generator(og_data_path, set_type=0))

    imcnn = SegImcnn(adapt_data=PartNetGraspDataset(og_data_path, set_type=0, only_signal=True))
    imcnn.load_state_dict(torch.load(model_path))
    classification_head = imcnn.model.output_dense

    # embs, labels, preds, map, pred_dict, g_unc_dict = get_embeddings_labels_preds_with_idx_map(og_dataset, imcnn,
    #                                                                                            classification_head,
    #                                                                                            range(0, 70))
    with open('/home/iroberts/projects/VisMeshSegmentation/cv_out_of_sample_preds.json', 'r') as f:
        cv_preds = json.load(f)

    # Convert to a dictionary for quick mesh_idx lookup
    cv_pred_dict = {entry['mesh_idx']: entry['preds'] for entry in cv_preds}

    # for i in range(0,70,20):
    embs, labels, preds, map, pred_dict, g_unc_dict = get_embeddings_labels_preds_with_idx_map(og_dataset, imcnn, classification_head,
                                                                            range(0, 70))
    all_preds = []
    for key in cv_pred_dict.keys():
        all_preds.append(cv_pred_dict[key])
    cv_all_preds = np.concatenate(all_preds, axis=0)

    pred_probs = sigmoid(cv_all_preds)
    issues = find_label_issues(
        labels=labels,
        pred_probs=pred_probs,
        return_indices_ranked_by="self_confidence",
    )
    print(issues)
    print(issues.shape)

    result = {}
    new_labels = {}

    for key, values in map.items():
        # values: list of original/global indices for this mesh
        # Find which indices in this mesh are considered issues
        issue_positions = [i for i, v in enumerate(values) if v in issues]

        # Extract original labels for this mesh
        original_labels = labels[values]  # these are the local labels for the mesh

        # Flip labels at issue positions: 1 becomes 0, 0 becomes 1
        flipped_labels = original_labels.copy()
        flipped_labels[issue_positions] = 1 - flipped_labels[issue_positions]

        # Store full label vector with flipped corrections
        new_labels[key] = flipped_labels

        # Also store the indices of the changes for later use
        if issue_positions:
            result[key] = issue_positions

    changed_idx_unc = {}
    for key in result.keys():
        changed_idx_unc[key] =[list(issues).index(val) for val in np.array(map[key])[result[key]]]

    sorted_by_unc, not_sorted_by_unc = global_mesh_sort(result, changed_idx_unc)

    path_to_corrections = "/home/iroberts/projects/VisMeshSegmentation/run_through/corrections/"
    os.mkdir(path_to_corrections + "confident_learning_")
    for i in list(range(500, 9500, 500)):
        if i > len(sorted_by_unc):
            file_name = "confident_learning_"+ "/confident_learning_"+ str(i) + ".csv"
            points_to_change = sorted_by_unc[:len(sorted_by_unc)]
            write_label_changes(path_to_corrections + file_name, points_to_change, new_labels)
        else:

            file_name = "confident_learning_" + "/confident_learning_"+ str(i) + ".csv"
            points_to_change = sorted_by_unc[:i]
            write_label_changes(path_to_corrections + file_name, points_to_change, new_labels)

    # # cleanlab works with **any classifier**. Yup, you can use PyTorch/TensorFlow/OpenAI/XGBoost/etc.
    # cl = cleanlab.classification.CleanLearning(classification_head)
    #
    # # cleanlab finds data and label issues in **any dataset**... in ONE line of code!
    # label_issues = cl.find_label_issues(embs, labels)
    #
    # # cleanlab trains a robust version of your model that works more reliably with noisy data.
    # cl.fit(embs, labels)

    # cleanlab estimates the predictions you would have gotten if you had trained with *no* label issues.
    # cl.predict(test_data)

    # A universal data-centric AI tool, cleanlab quantifies class-level issues and overall data quality, for any dataset.
    # cleanlab.dataset.health_summary(labels, confident_joint=cl.confident_joint)