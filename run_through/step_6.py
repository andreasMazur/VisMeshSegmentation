from improve_mesh_segmentation.data_correction.convert_partnet import convert_partnet
from run_through.step_3 import PARTNET_GRASP, DATASETS_PATH
from run_through.step_5 import CORRECTIONS_FILE


""" Step 6: Including the label corrections into the dataset

    After correcting, the corrected labels need to be incorporated into the originally preprocessed dataset. Run this
    script to include corrections into originally preprocessed dataset.
"""

PARTNET_GRASP_CORRECTED = f"{DATASETS_PATH}/partnet_grasp_corrected"
LABEL_CHANGES = f"{DATASETS_PATH}/label_changes"

if __name__ == "__main__":
    convert_partnet(
        old_data_path=PARTNET_GRASP,
        new_data_path=PARTNET_GRASP_CORRECTED,
        csv_path=CORRECTIONS_FILE,
        label_changes_path=LABEL_CHANGES  # Required for 'Step 8'
    )
