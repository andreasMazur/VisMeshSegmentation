from improve_mesh_segmentation.experiments.hypothesis_test import run_hypothesis_test
from run_through.step_3 import PARTNET_GRASP
from run_through.step_4 import LOGGING_DIR
from run_through.step_5 import CORRECTIONS_FILE
from run_through.step_6 import PARTNET_GRASP_CORRECTED


""" Step 7: Hypothesis test

    Run this script to redo the hypothesis test from the paper.
"""

if __name__ == "__main__":
    run_hypothesis_test(
        old_dataset_path=PARTNET_GRASP,
        new_dataset_path=PARTNET_GRASP_CORRECTED,
        csv_path=CORRECTIONS_FILE,
        logging_dir=f"{LOGGING_DIR}/hypothesis_test_logs",
        trials=30,
        epochs=10
    )
