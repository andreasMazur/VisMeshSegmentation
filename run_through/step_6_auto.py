import os

import numpy as np

from improve_mesh_segmentation.data_correction.convert_partnet import convert_partnet
from run_through.step_3 import DATASETS_PATH
from run_through.step_4 import PARTNET_GRASP
from run_through.step_5 import CORRECTIONS_FILE


""" Step 6: Including the label corrections into the dataset

    After correcting, the corrected labels need to be incorporated into the originally preprocessed dataset. Run this
    script to include corrections into originally preprocessed dataset.
"""

# PARTNET_GRASP_CORRECTED = f"{DATASETS_PATH}/partnet_grasp_corrected"
# LABEL_CHANGES = f"{DATASETS_PATH}/label_changes"

if __name__ == "__main__":
    corrections_path = "/home/iroberts/projects/VisMeshSegmentation/run_through/corrections/"
    # print(os.listdir(corrections_path))
    ks = np.arange(500, 9500, 500)
    # ['misclassification_sorted_unc', 'misclassification_sorted_inf', 'random_baseline', 'influence_baseline', "influence_uncertainty_combination_baseline",
    # "deepview_background_random", "deepview_background_sorted_inf", "deepview_background_sorted_unc","deepview_kmeans" ]

    # for j in list(range(1,11)):
    #     method = "global_lvq_"  + str(j)
    # js = [5,10,25,50,100,250,500]
    # for j in js:
    #     method = "global_cvknn_" + str(j)
    for method in ["cv_majority_baseline"]:
        for k in ks:
            PARTNET_GRASP_CORRECTED = "/home/iroberts/projects/VisMeshSegmentation/improve_mesh_segmentation/datasets/" + method + "/" + method + "_k_" + str(k)
            LABEL_CHANGES = None
            convert_partnet(
                    old_data_path=PARTNET_GRASP,
                    new_data_path=PARTNET_GRASP_CORRECTED,
                    csv_path=corrections_path + method + "/" + method + "_" + str(k)+".csv",
                    label_changes_path=LABEL_CHANGES  # Required for 'Step 8'
                )
