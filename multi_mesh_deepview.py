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
    influence_baseline, deepview_variants, influence_uncertainty_combination_baseline, deepview_kmeans, _lvq, \
    deepview_dbscan
from improve_mesh_segmentation.partnet_grasp.dataset import PartNetGraspDataset, processed_partnet_grasp_generator
from improve_mesh_segmentation.data_correction.correct_sub_partnet import embed
from improve_mesh_segmentation.training.imcnn import SegImcnn

from helper_functions import *

from sklearn.metrics import rand_score,adjusted_rand_score

from relabel_helpers.deepview_label_corrections import DeepViewLabelRevisit

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



    all_embeddings = []
    all_labels = []

    for mesh_idx, (((signal, bc), labels), ((_, _), cor_labels)) in enumerate(zip(og_dataset, corrected_dataset)):
        if mesh_idx <= 9:
            print(".....Gathering Data from Mesh Index:", mesh_idx)
            labels = np.array(labels)

            # Get embeddings for this mesh
            embeddings = embed(imcnn, [signal, bc])  # Assuming this returns a list or array of embeddings
            embeddings = torch.tensor(embeddings)  # Convert to torch tensor if it's not already

            all_embeddings.append(embeddings)
            all_labels.append(labels)

    # Concatenate all embeddings after the loop
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # --- Deep View Parameters ----
    batch_size = 32
    max_samples = 150000
    data_shape = (96,)
    resolution = 100
    N = 10
    lam = 1
    cmap = 'tab10'
    # to make shure deepview.show is blocking,
    # disable interactive mode
    interactive = False
    title = 'Automatic Relabeling'

    deepview = DeepViewLabelRevisit(pred_wrapper, np.arange(2), max_samples, batch_size, data_shape,
                                    N, lam, resolution, cmap, interactive, title, disc_dist=False)

    deepview.add_samples(all_embeddings, all_labels)
    print(deepview.distances.shape)
    # preds = np.argmax(classification_head(embeddings).detach().numpy(), axis=1)
