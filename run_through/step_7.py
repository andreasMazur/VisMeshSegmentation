from improve_mesh_segmentation.comparison_methods.confident_learning import confident_learning_partnet
from improve_mesh_segmentation.comparison_methods.ensemble_majority import ensemble_majority_partnet
from improve_mesh_segmentation.comparison_methods.global_methods import (
    global_kmeans_observed_partnet,
    global_knn_observed_partnet,
    global_lvq_observed_partnet,
    global_supervised_kmeans_observed_partnet,
)
from improve_mesh_segmentation.comparison_methods.incv import incv
from improve_mesh_segmentation.comparison_methods.local_methods import (
    local_kmeans_es_partnet,
    local_kmeans_umap_partnet,
    local_knn_es_partnet,
    local_knn_umap_partnet,
)
from improve_mesh_segmentation.comparison_methods.topofilter.run_topofilter_partnetgrasp import run_topofilter_experiments

from run_through.comparison_hyperparameters import (
    CV_EPOCHS,
    PARTNET_CL_CV_K,
    PARTNET_EM_CV_K,
    PARTNET_GLOBAL_KMEANS_PARAMETER,
    PARTNET_GLOBAL_KNN_PARAMETER,
    PARTNET_GLOBAL_LVQ_PARAMETER,
    PARTNET_GLOBAL_SUPERVISED_KMEANS_PARAMETER,
    PARTNET_LOCAL_KMEANS_ES_PARAMETER,
    PARTNET_LOCAL_KMEANS_UMAP_PARAMETER,
    PARTNET_LOCAL_KNN_ES_PARAMETER,
    PARTNET_LOCAL_KNN_UMAP_PARAMETER,
    TRAINING_EPOCHS,
)
from run_through.step_3 import PARTNET_GRASP, EXPERIMENT_DIRECTORY

import numpy as np
import os


""" Step 7: Run correction algorithms on PartNet-Grasp

    Hyperparameters for M_2–M_12: ``run_through/comparison_hyperparameters.py``
"""

if __name__ == "__main__":
    DATA_PATH = f"{PARTNET_GRASP}.zip"
    MODEL_PATH = f"{EXPERIMENT_DIRECTORY}/logs/model.zip"

    #################################
    # M_1: INCV
    # NOTE: INCV hyperparameters stay inline here (not in comparison_hyperparameters.py).
    #################################
    INCV_LOGS = f"{EXPERIMENT_DIRECTORY}/incv_logs"
    os.makedirs(INCV_LOGS, exist_ok=True)

    for training_epochs in range(1, 5):
        for max_iteration in range(1, 3):
            for remove_ratio in [0.1, 0.2, 0.3, 0.4, 0.5]:
                print(
                    f"\nCurrently running: INCV | "
                    f"Training epochs: {training_epochs} | "
                    f"Max iterations: {max_iteration} | "
                    f"Remove ratio: {remove_ratio}"
                )
                clean, correction_suggestions = incv(
                    data_path=DATA_PATH,
                    epochs=training_epochs,
                    max_iterations=max_iteration,
                    remove_ratio=remove_ratio,
                )
                np.save(
                    f"{INCV_LOGS}/incv_corrections_{training_epochs}_{max_iteration}_{remove_ratio}.npy",
                    correction_suggestions.numpy(),
                )

    #################################
    # M_2: Confident Learning
    #################################
    confident_learning_partnet(
        data_path=DATA_PATH,
        model_path=MODEL_PATH,
        experiment_directory=EXPERIMENT_DIRECTORY,
        cv_k=PARTNET_CL_CV_K,
        cv_epochs=CV_EPOCHS,
        training_epochs=TRAINING_EPOCHS,
    )

    #################################
    # M_3: TopoFilter
    #################################
    run_topofilter_experiments(
        model_type="all",
        partnet_grasp_path=DATA_PATH,
        correction_file_path=f"{EXPERIMENT_DIRECTORY}/data_correction/partnet_correction.csv",
        unq_v_triplets_path=f"{EXPERIMENT_DIRECTORY}/data_correction/unq_v_triplets.npy",
        results_path=f"{EXPERIMENT_DIRECTORY}/topofilter_results",
        start_cleans=[3],
        everys=[2],
        k_outliers=[64],
        k_ccs=[10],
        zetas=[0.5],
    )

    #################################
    # M_4: Ensemble Majority
    #################################
    ensemble_majority_partnet(
        data_path=DATA_PATH,
        experiment_directory=EXPERIMENT_DIRECTORY,
        cv_k=PARTNET_EM_CV_K,
        cv_epochs=CV_EPOCHS,
    )

    #################################
    # M_5: Global k-Means
    #################################
    if PARTNET_GLOBAL_KMEANS_PARAMETER is not None:
        global_kmeans_observed_partnet(
            data_path=DATA_PATH,
            model_path=MODEL_PATH,
            experiment_directory=EXPERIMENT_DIRECTORY,
            method_parameter=PARTNET_GLOBAL_KMEANS_PARAMETER,
            training_epochs=TRAINING_EPOCHS,
        )

    #################################
    # M_6: Global k-NN
    #################################
    if PARTNET_GLOBAL_KNN_PARAMETER is not None:
        global_knn_observed_partnet(
            data_path=DATA_PATH,
            model_path=MODEL_PATH,
            experiment_directory=EXPERIMENT_DIRECTORY,
            method_parameter=PARTNET_GLOBAL_KNN_PARAMETER,
            training_epochs=TRAINING_EPOCHS,
        )

    #################################
    # M_7: Global LVQ
    #################################
    if PARTNET_GLOBAL_LVQ_PARAMETER is not None:
        global_lvq_observed_partnet(
            data_path=DATA_PATH,
            model_path=MODEL_PATH,
            experiment_directory=EXPERIMENT_DIRECTORY,
            method_parameter=PARTNET_GLOBAL_LVQ_PARAMETER,
            training_epochs=TRAINING_EPOCHS,
        )

    #################################
    # M_8: Global supervised k-Means
    #################################
    if PARTNET_GLOBAL_SUPERVISED_KMEANS_PARAMETER is not None:
        global_supervised_kmeans_observed_partnet(
            data_path=DATA_PATH,
            model_path=MODEL_PATH,
            experiment_directory=EXPERIMENT_DIRECTORY,
            method_parameter=PARTNET_GLOBAL_SUPERVISED_KMEANS_PARAMETER,
            training_epochs=TRAINING_EPOCHS,
        )

    #################################
    # M_9: Local k-Means (UMAP via DeepView)
    #################################
    if PARTNET_LOCAL_KMEANS_UMAP_PARAMETER is not None:
        local_kmeans_umap_partnet(
            data_path=DATA_PATH,
            model_path=MODEL_PATH,
            experiment_directory=EXPERIMENT_DIRECTORY,
            method_parameter=PARTNET_LOCAL_KMEANS_UMAP_PARAMETER,
            training_epochs=TRAINING_EPOCHS,
        )

    #################################
    # M_10: Local k-NN (UMAP via DeepView)
    #################################
    if PARTNET_LOCAL_KNN_UMAP_PARAMETER is not None:
        local_knn_umap_partnet(
            data_path=DATA_PATH,
            model_path=MODEL_PATH,
            experiment_directory=EXPERIMENT_DIRECTORY,
            method_parameter=PARTNET_LOCAL_KNN_UMAP_PARAMETER,
            training_epochs=TRAINING_EPOCHS,
        )

    #################################
    # M_11: Local k-Means (ES)
    #################################
    if PARTNET_LOCAL_KMEANS_ES_PARAMETER is not None:
        local_kmeans_es_partnet(
            data_path=DATA_PATH,
            model_path=MODEL_PATH,
            experiment_directory=EXPERIMENT_DIRECTORY,
            method_parameter=PARTNET_LOCAL_KMEANS_ES_PARAMETER,
            training_epochs=TRAINING_EPOCHS,
        )

    #################################
    # M_12: k-NN (ES)
    #################################
    if PARTNET_LOCAL_KNN_ES_PARAMETER is not None:
        local_knn_es_partnet(
            data_path=DATA_PATH,
            model_path=MODEL_PATH,
            experiment_directory=EXPERIMENT_DIRECTORY,
            method_parameter=PARTNET_LOCAL_KNN_ES_PARAMETER,
            training_epochs=TRAINING_EPOCHS,
        )
