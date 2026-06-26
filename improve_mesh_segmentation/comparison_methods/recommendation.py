import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def recommend_knn_binarized(dists, true_labs, reference_labels, n_neighbors=5, recommendation_percentage=100):
    nn = KNeighborsClassifier(n_neighbors=n_neighbors)
    nn.fit(dists, true_labs)

    neighs = nn.kneighbors(return_distance=False)
    neigh_labs = reference_labels[neighs]

    n_half = int(n_neighbors / 2)
    mask = np.sum(neigh_labs == true_labs[:, None], axis=1)
    flips = mask <= n_half
    mismatch_indices = np.where(flips)[0]

    final_labs = np.zeros_like(true_labs, dtype=int)
    final_labs[mismatch_indices] = 1

    return final_labs, mismatch_indices


def recommend_knn_based(dists, true_labs, reference_labels, n_neighbors=5, recommendation_percentage=100):
    nn = KNeighborsClassifier(n_neighbors=n_neighbors)
    nn.fit(dists, true_labs)
    unique_l = np.unique(true_labs)

    neighs = nn.kneighbors(return_distance=False)
    neigh_labs = reference_labels[neighs]

    counts_cl = np.zeros([true_labs.shape[0], unique_l.shape[0]])
    for i in range(unique_l.shape[0]):
        counts_cl[:, i] = np.sum(neigh_labs == unique_l[i], axis=1)

    pred_labs = unique_l[counts_cl.argmax(axis=1)]
    mismatch_indices = np.where(pred_labs != true_labs)[0]

    recommend = recommendation_percentage / 100
    num_recommendations = int(recommend * len(mismatch_indices))
    keep_indices = np.random.choice(mismatch_indices, num_recommendations, replace=False)

    final_labs = true_labs.copy()
    final_labs[keep_indices] = pred_labs[keep_indices]

    return final_labs, mismatch_indices, keep_indices
