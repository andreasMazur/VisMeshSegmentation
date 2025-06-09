import os

from improve_mesh_segmentation.training.train_imcnn import train_single_imcnn, train_imcnn_cv
from run_through.step_3 import EXPERIMENT_DIRECTORY, PARTNET_GRASP
from sklearn.model_selection import KFold
import numpy as np

""" Step 4: Train an initial IMCNN

    A trained IMCNN is required for the label correction process. By running this script, you train an IMCNN.
"""


LOGGING_DIR = f"{EXPERIMENT_DIRECTORY}/logs/cv/"
PARTNET_GRASP = "/home/iroberts/projects/VisMeshSegmentation/improve_mesh_segmentation/datasets/partnet_grasp.zip"
if __name__ == "__main__":



    model_dirs = os.listdir("/home/iroberts/projects/VisMeshSegmentation/run_through/logs/cv")

    for model in model_dirs:
        imcnn = SegImcnn(adapt_data=PartNetGraspDataset(og_data_path, set_type=0, only_signal=True))
        imcnn.load_state_dict(torch.load(model_path))
        classification_head = imcnn.model.output_dense
        og_dataset = list(processed_partnet_grasp_generator(og_data_path, set_type=0))
    # train_imcnn_cv(
    #     data_path=PARTNET_GRASP,
    #     n_epochs=10,
    #     logging_dir=LOGGING_DIR,
    #     skip_validation=True,
    #     skip_testing=False,
    #     verbose=True
    # )
