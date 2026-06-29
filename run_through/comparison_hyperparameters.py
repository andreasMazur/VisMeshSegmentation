"""Hyperparameters for comparison methods M_2–M_12 (step 7 / step 8).

INCV (M_1) hyperparameters stay inline in the step scripts.
Edit values here to match the paper / hyperparameter search.
"""

# ---------------------------------------------------------------------------
# Shared training
# ---------------------------------------------------------------------------
CV_EPOCHS = 10
TRAINING_EPOCHS = 10  # step-4 IMCNN epochs if model.zip is missing

# ---------------------------------------------------------------------------
# PartNet-Grasp (step 7): M_2–M_12
# ---------------------------------------------------------------------------
PARTNET_CL_CV_K = 5   # M_2 Confident Learning
PARTNET_EM_CV_K = 25  # M_4 Ensemble Majority

PARTNET_GLOBAL_KMEANS_PARAMETER = 6
PARTNET_GLOBAL_KNN_PARAMETER = 2000
PARTNET_GLOBAL_LVQ_PARAMETER = 4
PARTNET_GLOBAL_SUPERVISED_KMEANS_PARAMETER = 2

PARTNET_LOCAL_KMEANS_UMAP_PARAMETER = 50  # DeepView k-Means (M_9)
PARTNET_LOCAL_KNN_UMAP_PARAMETER = 5       # DeepView k-NN (M_10)
PARTNET_LOCAL_KMEANS_ES_PARAMETER = 5
PARTNET_LOCAL_KNN_ES_PARAMETER = 50

# ---------------------------------------------------------------------------
# FAUST (step 8): M_2–M_12  (per noise level)
# ---------------------------------------------------------------------------
FAUST_CL_CV_K = 25
FAUST_EM_CV_K = 20

FAUST_PARAMETERS = {
    "low_noise": {
        "global_kmeans": 16,
        "global_knn": 5,
        "global_lvq": 5,
        "global_supervised_kmeans": 5,
        "local_kmeans_umap": 11,
        "local_kmeans_umap_n_neighbors": 50,
        "local_kmeans_umap_n_components": 10,
        "local_knn_umap": 5,
        "local_knn_umap_n_neighbors": 5,
        "local_knn_umap_n_components": 5,
        "local_kmeans_es": 11,
        "local_knn_es": 50,
        "topofilter_params": {
            "start_cleans": [4],
            "everys": [3],
            "k_ccs": [10],
            "k_outliers": [32],
            "zetas": [0.25],
        },
    },
    "mid_noise": {
        "global_kmeans": 8,
        "global_knn": 25,
        "global_lvq": 4,
        "global_supervised_kmeans": 1,
        "local_kmeans_umap": 11,
        "local_kmeans_umap_n_neighbors": 25,
        "local_kmeans_umap_n_components": 4,
        "local_knn_umap": 25,
        "local_knn_umap_n_neighbors": 5,
        "local_knn_umap_n_components": 10,
        "local_kmeans_es": 11,
        "local_knn_es": 10,
        "topofilter_params": {
            "start_cleans": [3],
            "everys": [3],
            "k_ccs": [10],
            "k_outliers": [32],
            "zetas": [0.5],
        },
    },
    "high_noise": {
        "global_kmeans": 9,
        "global_knn": 5,
        "global_lvq": 4,
        "global_supervised_kmeans": 1,
        "local_kmeans_umap": 11,
        "local_kmeans_umap_n_neighbors": 5,
        "local_kmeans_umap_n_components": 4,
        "local_knn_umap": 5,
        "local_knn_umap_n_neighbors": 5,
        "local_knn_umap_n_components": 4,
        "local_kmeans_es": 11,
        "local_knn_es": 50,
        "topofilter_params": {
            "start_cleans": [2],
            "everys": [2],
            "k_ccs": [10],
            "k_outliers": [32],
            "zetas": [0.25],
        },
    },
}

PARTNET_GRASP_PARAMETERS = {
    "low_noise": {
        "topofilter_params": {
            "start_cleans": [4],
            "everys": [3],
            "k_ccs": [10],
            "k_outliers": [32],
            "zetas": [0.25],
        },
    },
    "mid_noise": {
        "topofilter_params": {
            "start_cleans": [3],
            "everys": [3],
            "k_ccs": [10],
            "k_outliers": [32],
            "zetas": [0.5],
        },
    },
    "high_noise": {
        "topofilter_params": {
            "start_cleans": [2],
            "everys": [2],
            "k_ccs": [10],
            "k_outliers": [32],
            "zetas": [0.25],
        },
    },
}
