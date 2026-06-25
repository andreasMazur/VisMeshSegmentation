from improve_mesh_segmentation.comparison_methods.incv import incv

from run_through.step_3 import EXPERIMENT_DIRECTORY

import numpy as np
import os


""" Step 8: Run correction algorithms on the FAUST segmentation dataset.

"""

if __name__ == "__main__":
    # Define dataset paths
    faust_root = "PATH/TO/NOISY/FAUST/DIRECTORY"
    noise_ds_1 = (f"{faust_root}/faust_segmentation_logs_noise_lvl_0.007", "faust_low_noise")
    noise_ds_2 = (f"{faust_root}/faust_segmentation_logs_noise_lvl_0.036", "faust_mid_noise")
    noise_ds_3 = (f"{faust_root}/faust_segmentation_logs_noise_lvl_0.071", "faust_high_noise")

    for (dataset_path, dataset_id) in [noise_ds_1, noise_ds_2, noise_ds_3]:
        #################################
        # M_1: INCV
        #################################
        INCV_LOGS = f"{EXPERIMENT_DIRECTORY}/{dataset_id}/incv_logs"
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
                        data_path=f"{dataset_path}.zip",
                        epochs=training_epochs,
                        max_iterations=max_iteration
                    )
                    np.save(
                        f"{INCV_LOGS}/incv_corrections_{training_epochs}_{max_iteration}_{remove_ratio}.npy",
                        correction_suggestions.numpy()
                    )

        #################################
        # M_2: Confident Learning
        #################################
        pass  # TODO

        #################################
        # M_3: TopoFilter
        #################################
        pass  # TODO

        #################################
        # M_4: Ensemble Majority
        #################################
        pass  # TODO

        #################################
        # M_5: Global k-Means
        #################################
        pass  # TODO

        #################################
        # M_6: Global k-NN
        #################################
        pass  # TODO

        #################################
        # M_7: Global LVQ
        #################################
        pass  # TODO

        #################################
        # M_8: Global supervised k-Means
        #################################
        pass  # TODO

        #################################
        # M_9: Local k-Means (UMAP)
        #################################
        pass  # TODO

        #################################
        # M_10: Local k-NN (UMAP)
        #################################
        pass  # TODO

        #################################
        # M_11: Local k-Means (ES)
        #################################
        pass  # TODO

        #################################
        # M_12: k-NN (ES)
        #################################
        pass  # TODO
