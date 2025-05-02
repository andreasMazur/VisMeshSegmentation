from improve_mesh_segmentation.training.train_imcnn import train_single_imcnn
from run_through.step_3 import EXPERIMENT_DIRECTORY, PARTNET_GRASP


""" Step 4: Train an initial IMCNN

    A trained IMCNN is required for the label correction process. By running this script, you train an IMCNN.
"""


LOGGING_DIR = f"{EXPERIMENT_DIRECTORY}/logs"
PARTNET_GRASP = "/home/iroberts/projects/VisMeshSegmentation/improve_mesh_segmentation/datasets/partnet_grasp"
if __name__ == "__main__":
    train_single_imcnn(
        data_path=PARTNET_GRASP,
        n_epochs=10,
        logging_dir=LOGGING_DIR,
        skip_validation=False,
        skip_testing=False,
        verbose=True
    )
