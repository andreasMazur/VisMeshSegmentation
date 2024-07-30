from improve_mesh_segmentation.data_correction.correct_sub_partnet import correct_sub_partnet
from run_through.step_3 import PARTNET_GRASP, EXPERIMENT_DIRECTORY, DATASETS_PATH
from run_through.step_4 import LOGGING_DIR

import matplotlib


""" Step 5: Correcting segmentation labels

    Now the noisy segmentation labels from the alignment process need to be corrected. There are two options for this.
    - One can either use our pre-made corrections by referring to the correction file within this repository (OPTION 1).
    - Alternatively, one can run the segmentation label correction algorithm to interactively correct segmentation
      labels of PartNet-Grasp (OPTION 2).
    Lastly, execute this script to start the correction algorithm.
"""

# TODO: Select 'OPTION 1 = True' or 'OPTION 2 = False'.
USE_PREDEFINED_CORRECTIONS = True

# Use this corrections file if you want to use pre-made correction suggestions
if USE_PREDEFINED_CORRECTIONS == 1:
    CORRECTIONS_FILE = f"{EXPERIMENT_DIRECTORY}/improve_mesh_segmentation/data_correction/partnet_correction.csv"
else:
    CORRECTIONS_FILE = f"{DATASETS_PATH}/partnet_correction.csv"

# You only need to run this if you are NOT using 'OPTION 1'.
if __name__ == "__main__":
    matplotlib.use("Qt5Agg")
    correct_sub_partnet(
        data_path=f"{PARTNET_GRASP}.zip",
        model_path=f"{LOGGING_DIR}/model.zip",
        correction_csv_path=CORRECTIONS_FILE
    )
