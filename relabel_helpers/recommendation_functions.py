import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import json

def recommend_KNN_based(dists, true_labs, reference_labels, n_neighbors=5, recommendation_percentage=100):
    nn = KNeighborsClassifier(n_neighbors=n_neighbors)
    nn.fit(dists, true_labs)
    unique_l = np.unique(true_labs)

    # Get neighbors and their labels
    neighs = nn.kneighbors(return_distance=False)
    neigh_labs = reference_labels[neighs]

    # Count occurrences of each class in neighbors
    counts_cl = np.zeros([true_labs.shape[0], unique_l.shape[0]])
    for i in range(unique_l.shape[0]):
        counts_cl[:, i] = np.sum(neigh_labs == unique_l[i], axis=1)

    pred_labs = unique_l[counts_cl.argmax(axis=1)]
    mismatch_indices = np.where(pred_labs != true_labs)[0]

    # Decide how many recommendations to keep
    recommend = recommendation_percentage / 100
    num_indices = len(mismatch_indices)
    num_recommendations = int(recommend * num_indices)

    # Randomly select indices to accept recommendations
    keep_indices = np.random.choice(mismatch_indices, num_recommendations, replace=False)

    # Create final recommended labels: start with true labels
    final_labs = true_labs.copy()
    final_labs[keep_indices] = pred_labs[keep_indices]

    return final_labs, mismatch_indices, keep_indices


def recommend_KNN_w_influence_based(dists, true_labs, reference_labels, mesh_idx, n_neighbors=5, recommendation_percentage=100):
    nn = KNeighborsClassifier(n_neighbors=n_neighbors)
    nn.fit(dists, true_labs)
    unique_l = np.unique(true_labs)

    # Get neighbors and their labels
    neighs = nn.kneighbors(return_distance=False)
    neigh_labs = reference_labels[neighs]

    # Count occurrences of each class in neighbors
    counts_cl = np.zeros([true_labs.shape[0], unique_l.shape[0]])
    for i in range(unique_l.shape[0]):
        counts_cl[:, i] = np.sum(neigh_labs == unique_l[i], axis=1)

    pred_labs = unique_l[counts_cl.argmax(axis=1)]
    mismatch_indices = np.where(pred_labs != true_labs)[0]

    # Decide how many recommendations to keep
    # recommend = recommendation_percentage / 100
    # num_indices = len(mismatch_indices)
    # num_recommendations = int(recommend * num_indices)

    # Randomly select indices to accept recommendations
    with open('/home/iroberts/projects/VisMeshSegmentation/influence_scores.json', 'r') as f:
        influence_scores = json.load(f)

    influence_dict = {entry['mesh_idx']: entry['influence_per_vertex'] for entry in influence_scores}

    influential_mismatch_idx = []
    for mismatch in mismatch_indices:
        if abs(influence_dict[mesh_idx][mismatch]) > (7.565/2): # 7.565 is global std of influences
            influential_mismatch_idx.append(mismatch)

    recommend = recommendation_percentage / 100
    num_indices = len(influential_mismatch_idx)
    num_recommendations = int(recommend * num_indices)

    keep_indices = np.random.choice(influential_mismatch_idx, num_recommendations, replace=False)

    if len(keep_indices) > 0:

        # Create final recommended labels: start with true labels
        final_labs = true_labs.copy()
        final_labs[keep_indices] = pred_labs[keep_indices]

    else:
        final_labs = true_labs.copy()
    return final_labs, influential_mismatch_idx, keep_indices


def recommend_unc_based(preds, true_labs, unc, n_neighbors=5, recommendation_percentage=100):
    nn = KNeighborsClassifier(n_neighbors=n_neighbors)
    nn.fit(dists, true_labs)
    unique_l = np.unique(true_labs)

    # Get neighbors and their labels
    neighs = nn.kneighbors(return_distance=False)
    neigh_labs = reference_labels[neighs]

    # Count occurrences of each class in neighbors
    counts_cl = np.zeros([true_labs.shape[0], unique_l.shape[0]])
    for i in range(unique_l.shape[0]):
        counts_cl[:, i] = np.sum(neigh_labs == unique_l[i], axis=1)

    pred_labs = unique_l[counts_cl.argmax(axis=1)]
    mismatch_indices = np.where(pred_labs != true_labs)[0]

    # Decide how many recommendations to keep
    recommend = recommendation_percentage / 100
    num_indices = len(mismatch_indices)
    num_recommendations = int(recommend * num_indices)

    # Randomly select indices to accept recommendations
    keep_indices = np.random.choice(mismatch_indices, num_recommendations, replace=False)

    # Create final recommended labels: start with true labels
    final_labs = true_labs.copy()
    final_labs[keep_indices] = pred_labs[keep_indices]

    return final_labs, mismatch_indices, keep_indices
