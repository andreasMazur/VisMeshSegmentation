import os

from improve_mesh_segmentation.partnet_grasp.dataset import PartNetGraspDataset
from improve_mesh_segmentation.training.imcnn import SegImcnn
from improve_mesh_segmentation.training.train_imcnn import train_single_imcnn, train_imcnn_cv
from run_through.step_3 import EXPERIMENT_DIRECTORY, PARTNET_GRASP
from sklearn.model_selection import KFold
import numpy as np

""" Step 4: Train an initial IMCNN

    A trained IMCNN is required for the label correction process. By running this script, you train an IMCNN.
"""


LOGGING_DIR = f"{EXPERIMENT_DIRECTORY}/logs/behanvior_exp/cv/"
PARTNET_GRASP = "/home/iroberts/projects/VisMeshSegmentation/improve_mesh_segmentation/datasets/partnet_grasp.zip"
if __name__ == "__main__":
    folds = [2,4,5]
    for fold in folds:
        train_imcnn_cv(
            data_path=PARTNET_GRASP,
            n_epochs=10,
            K = fold,
            logging_dir=LOGGING_DIR+f"{fold}_fold_cv",
            skip_validation=True,
            skip_testing=False,
            verbose=True
        )
