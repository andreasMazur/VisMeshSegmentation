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
    # correction_percentages = np.arange(10, 110, 10)
    # for percent in correction_percentages :
    PARTNET_GRASP_CORRECTED = ("/home/iroberts/projects/VisMeshSegmentation/improve_mesh_segmentation/datasets/partnet_grasp_corrected_deepview_influence_background_"
                                   + str(100))
    LABEL_CHANGES = None
    convert_partnet(
            old_data_path=PARTNET_GRASP,
            new_data_path=PARTNET_GRASP_CORRECTED,
            csv_path=corrections_path + "corrected_labels_deepview_influence_background_"+str(100)+".csv",
            label_changes_path=LABEL_CHANGES  # Required for 'Step 8'
        )
