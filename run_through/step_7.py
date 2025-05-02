import os

from improve_mesh_segmentation.experiments.hypothesis_test import run_hypothesis_test,run_graph
from run_through.step_3 import PARTNET_GRASP
from run_through.step_4 import LOGGING_DIR
from run_through.step_5 import CORRECTIONS_FILE
from run_through.step_6 import PARTNET_GRASP_CORRECTED


""" Step 7: Hypothesis test

    Run this script to redo the hypothesis test from the paper.
"""

if __name__ == "__main__":
    datasets_path = "/home/iroberts/projects/VisMeshSegmentation/improve_mesh_segmentation/datasets/"
    datasets = ['partnet_grasp.zip','partnet_grasp_corrected_deepviewbackground_10.zip', 'partnet_grasp_corrected_deepviewbackground_20.zip',
     'partnet_grasp_corrected_deepviewbackground_30.zip', 'partnet_grasp_corrected_deepviewbackground_40.zip',
     'partnet_grasp_corrected_deepviewbackground_50.zip', 'partnet_grasp_corrected_deepviewbackground_60.zip',
     'partnet_grasp_corrected_deepviewbackground_70.zip', 'partnet_grasp_corrected_deepviewbackground_80.zip',
     'partnet_grasp_corrected_deepviewbackground_90.zip', 'partnet_grasp_corrected_deepviewbackground_100.zip',
     'partnet_grasp_corrected.zip',]

    # print(os.listdir(datasets_path))
    # print([datasets_path + dataset for dataset in datasets])
    run_graph(
        paths=[datasets_path + dataset for dataset in datasets],
        logging_dir=f"{LOGGING_DIR}/hypothesis_test_logs",
        trials=30,
        epochs=10
    )
