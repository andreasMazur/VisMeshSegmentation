from improve_mesh_segmentation.data_correction.correct_sub_partnet import correct_sub_partnet
from run_through.step_3 import PARTNET_GRASP, EXPERIMENT_DIRECTORY, DATASETS_PATH
from run_through.step_4 import LOGGING_DIR

import matplotlib


# OPTION 1:
# Use this corrections file if you want to use pre-made correction suggestions
CORRECTIONS_FILE = f"{EXPERIMENT_DIRECTORY}/improve_mesh_segmentation/data_correction/partnet_correction.csv"

# OPTION 2:
# Comment above 'CORRECTIONS_FILE' and use a different filepath if you want to correct labels yourself.
# CORRECTIONS_FILE = f"{DATASETS_PATH}/partnet_correction.csv"

# You only need to run this if you are NOT using 'OPTION 1'.
if __name__ == "__main__":
    matplotlib.use("Qt5Agg")
    correct_sub_partnet(
        data_path=f"{PARTNET_GRASP}.zip",
        model_path=f"{LOGGING_DIR}/model.zip",
        correction_csv_path=CORRECTIONS_FILE
    )
