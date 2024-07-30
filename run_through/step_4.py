from improve_mesh_segmentation.training.train_imcnn import train_single_imcnn
from run_through.step_3 import EXPERIMENT_DIRECTORY, PARTNET_GRASP


""" Step 4: Train an initial IMCNN

    A trained IMCNN is required for the label correction process. By running this script, you train an IMCNN.
"""


LOGGING_DIR = f"{EXPERIMENT_DIRECTORY}/logs"

if __name__ == "__main__":
    train_single_imcnn(
        data_path=f"{PARTNET_GRASP}.zip",
        n_epochs=10,
        logging_dir=LOGGING_DIR,
        skip_validation=False,
        skip_testing=False,
        verbose=True
    )
