from improve_mesh_segmentation.comparison_methods.confident_learning import confident_learning_faust
from improve_mesh_segmentation.comparison_methods.ensemble_majority import ensemble_majority_faust
from improve_mesh_segmentation.comparison_methods.global_methods import (
    global_kmeans_observed_faust,
    global_knn_observed_faust,
    global_lvq_observed_faust,
    global_supervised_kmeans_observed_faust,
)
from improve_mesh_segmentation.comparison_methods.incv import incv
from improve_mesh_segmentation.comparison_methods.local_methods import (
    local_kmeans_es_faust,
    local_kmeans_umap_faust,
    local_knn_es_faust,
    local_knn_umap_faust,
)

from run_through.comparison_hyperparameters import (
    CV_EPOCHS,
    FAUST_CL_CV_K,
    FAUST_EM_CV_K,
    FAUST_PARAMETERS,
    TRAINING_EPOCHS,
)
from run_through.step_3 import EXPERIMENT_DIRECTORY

import numpy as np
import os


""" Step 8: Run correction algorithms on the FAUST segmentation dataset.

    Hyperparameters for M_2–M_12: ``run_through/comparison_hyperparameters.py``
"""

if __name__ == "__main__":
    FAUST_ROOT = "PATH/TO/NOISY/FAUST/DIRECTORY"
    FAUST_ZIP = "PATH/TO/faust_preprocess.zip"
    SEGMENTATION_LABELS = "PATH/TO/segmentation_labels.npy"

    noise_ds_1 = (f"{FAUST_ROOT}/faust_segmentation_logs_noise_lvl_0.007", "faust_low_noise", 0.007)
    noise_ds_2 = (f"{FAUST_ROOT}/faust_segmentation_logs_noise_lvl_0.036", "faust_mid_noise", 0.036)
    noise_ds_3 = (f"{FAUST_ROOT}/faust_segmentation_logs_noise_lvl_0.071", "faust_high_noise", 0.071)

    for (logging_dir, dataset_id, noise_level) in [noise_ds_1, noise_ds_2, noise_ds_3]:
        dataset_path = f"{logging_dir}.zip"
        model_path = f"{logging_dir}/model.zip"
        params = FAUST_PARAMETERS[dataset_id]

        #################################
        # M_1: INCV
        # NOTE: INCV hyperparameters stay inline here (not in comparison_hyperparameters.py).
        # NOTE: FAUST INCV still uses the PartNet incv() entry point — wire FAUST incv separately.
        #################################
        INCV_LOGS = f"{EXPERIMENT_DIRECTORY}/{dataset_id}/incv_logs"
        os.makedirs(INCV_LOGS, exist_ok=True)

        for training_epochs in range(1, 5):
            for max_iteration in range(1, 3):
                for remove_ratio in [0.1, 0.2, 0.3, 0.4, 0.5]:
                    print(
                        f"\nCurrently running: INCV | Dataset: {dataset_id} | "
                        f"Training epochs: {training_epochs} | "
                        f"Max iterations: {max_iteration} | "
                        f"Remove ratio: {remove_ratio}"
                    )
                    clean, correction_suggestions = incv(
                        data_path=dataset_path,
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
        confident_learning_faust(
            dataset_path=FAUST_ZIP,
            segmentation_labels_path=SEGMENTATION_LABELS,
            logging_dir=logging_dir,
            noise_level=noise_level,
            model_path=model_path,
            experiment_directory=EXPERIMENT_DIRECTORY,
            dataset_id=dataset_id,
            cv_k=FAUST_CL_CV_K,
            cv_epochs=CV_EPOCHS,
            training_epochs=TRAINING_EPOCHS,
        )

        #################################
        # M_3: TopoFilter (colleague's responsibility)
        #################################
        pass

        #################################
        # M_4: Ensemble Majority
        #################################
        ensemble_majority_faust(
            dataset_path=FAUST_ZIP,
            segmentation_labels_path=SEGMENTATION_LABELS,
            logging_dir=logging_dir,
            noise_level=noise_level,
            experiment_directory=EXPERIMENT_DIRECTORY,
            dataset_id=dataset_id,
            cv_k=FAUST_EM_CV_K,
            cv_epochs=CV_EPOCHS,
        )

        #################################
        # M_5: Global k-Means
        #################################
        if params["global_kmeans"] is not None:
            global_kmeans_observed_faust(
                dataset_path=FAUST_ZIP,
                segmentation_labels_path=SEGMENTATION_LABELS,
                logging_dir=logging_dir,
                noise_level=noise_level,
                model_path=model_path,
                experiment_directory=EXPERIMENT_DIRECTORY,
                dataset_id=dataset_id,
                method_parameter=params["global_kmeans"],
                training_epochs=TRAINING_EPOCHS,
            )

        #################################
        # M_6: Global k-NN
        #################################
        if params["global_knn"] is not None:
            global_knn_observed_faust(
                dataset_path=FAUST_ZIP,
                segmentation_labels_path=SEGMENTATION_LABELS,
                logging_dir=logging_dir,
                noise_level=noise_level,
                model_path=model_path,
                experiment_directory=EXPERIMENT_DIRECTORY,
                dataset_id=dataset_id,
                method_parameter=params["global_knn"],
                training_epochs=TRAINING_EPOCHS,
            )

        #################################
        # M_7: Global LVQ
        #################################
        if params["global_lvq"] is not None:
            global_lvq_observed_faust(
                dataset_path=FAUST_ZIP,
                segmentation_labels_path=SEGMENTATION_LABELS,
                logging_dir=logging_dir,
                noise_level=noise_level,
                model_path=model_path,
                experiment_directory=EXPERIMENT_DIRECTORY,
                dataset_id=dataset_id,
                method_parameter=params["global_lvq"],
                training_epochs=TRAINING_EPOCHS,
            )

        #################################
        # M_8: Global supervised k-Means
        #################################
        if params["global_supervised_kmeans"] is not None:
            global_supervised_kmeans_observed_faust(
                dataset_path=FAUST_ZIP,
                segmentation_labels_path=SEGMENTATION_LABELS,
                logging_dir=logging_dir,
                noise_level=noise_level,
                model_path=model_path,
                experiment_directory=EXPERIMENT_DIRECTORY,
                dataset_id=dataset_id,
                method_parameter=params["global_supervised_kmeans"],
                training_epochs=TRAINING_EPOCHS,
            )

        #################################
        # M_9: Local k-Means (UMAP)
        #################################
        if (
            params["local_kmeans_umap"] is not None
            and params["local_kmeans_umap_n_neighbors"] is not None
            and params["local_kmeans_umap_n_components"] is not None
        ):
            local_kmeans_umap_faust(
                dataset_path=FAUST_ZIP,
                segmentation_labels_path=SEGMENTATION_LABELS,
                logging_dir=logging_dir,
                noise_level=noise_level,
                model_path=model_path,
                experiment_directory=EXPERIMENT_DIRECTORY,
                dataset_id=dataset_id,
                method_parameter=(
                    params["local_kmeans_umap"],
                    params["local_kmeans_umap_n_neighbors"],
                    params["local_kmeans_umap_n_components"],
                ),
                training_epochs=TRAINING_EPOCHS,
            )

        #################################
        # M_10: Local k-NN (UMAP)
        #################################
        if (
            params["local_knn_umap"] is not None
            and params["local_knn_umap_n_neighbors"] is not None
            and params["local_knn_umap_n_components"] is not None
        ):
            local_knn_umap_faust(
                dataset_path=FAUST_ZIP,
                segmentation_labels_path=SEGMENTATION_LABELS,
                logging_dir=logging_dir,
                noise_level=noise_level,
                model_path=model_path,
                experiment_directory=EXPERIMENT_DIRECTORY,
                dataset_id=dataset_id,
                method_parameter=(
                    params["local_knn_umap"],
                    params["local_knn_umap_n_neighbors"],
                    params["local_knn_umap_n_components"],
                ),
                training_epochs=TRAINING_EPOCHS,
            )

        #################################
        # M_11: Local k-Means (ES)
        #################################
        if params["local_kmeans_es"] is not None:
            local_kmeans_es_faust(
                dataset_path=FAUST_ZIP,
                segmentation_labels_path=SEGMENTATION_LABELS,
                logging_dir=logging_dir,
                noise_level=noise_level,
                model_path=model_path,
                experiment_directory=EXPERIMENT_DIRECTORY,
                dataset_id=dataset_id,
                method_parameter=params["local_kmeans_es"],
                training_epochs=TRAINING_EPOCHS,
            )

        #################################
        # M_12: k-NN (ES)
        #################################
        if params["local_knn_es"] is not None:
            local_knn_es_faust(
                dataset_path=FAUST_ZIP,
                segmentation_labels_path=SEGMENTATION_LABELS,
                logging_dir=logging_dir,
                noise_level=noise_level,
                model_path=model_path,
                experiment_directory=EXPERIMENT_DIRECTORY,
                dataset_id=dataset_id,
                method_parameter=params["local_knn_es"],
                training_epochs=TRAINING_EPOCHS,
            )
