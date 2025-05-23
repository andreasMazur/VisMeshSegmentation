from matplotlib import pyplot as plt

from improve_mesh_segmentation.partnet_grasp.dataset import PartNetGraspDataset
from improve_mesh_segmentation.data_correction.convert_partnet import convert_partnet
from improve_mesh_segmentation.training.train_imcnn import train_single_imcnn

from pathlib import Path

import os
import scipy as sp
import numpy as np


def run_hypothesis_test(old_dataset_path,
                        new_dataset_path,
                        csv_path,
                        logging_dir,
                        trials=30,
                        epochs=10):
    """Creates the datasets and starts the training runs.

    Parameters
    ----------
    old_dataset_path: str
        The path to the originally pre-processed FAUST dataset
    csv_path: str
        The path to the CSV-file that contains the DeepView-corrections
    new_dataset_path: str
        The path to the DeepView-corrected dataset
    logging_dir: str
        The path to the logging directory
    trials: int
        The amount of trials (i.e. amount of training runs for each corrected and uncorrected sub-experiment)
    epochs: int
        The amount of epochs per trial
    """
    # Create logging dir
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)

    # Integrate DeepView-corrected labels into the dataset
    new_dataset_path = f"{new_dataset_path}.zip"
    if not Path(new_dataset_path).is_file():
        convert_partnet(old_data_path=old_dataset_path, new_data_path=new_dataset_path, csv_path=csv_path)
    else:
        print(f"Found converted dataset file: '{new_dataset_path}'. Skipping preprocessing.")
    old_dataset_path = f"{old_dataset_path}.zip"

    # Start training runs
    test_accuracies, test_losses, sub_logging_dirs = [[], []], [[], []], ["bbox_approach", "deepview_approach"]
    for idx, zip_file in enumerate([old_dataset_path, new_dataset_path]):
        # Train, validate and test IMCNN
        for trial_idx in range(trials):
            print(f"Using un-corrected data: {idx == 0} | Using corrected data: {idx == 1} | Trial {trial_idx}")
            adaptation_data = PartNetGraspDataset(zip_file, set_type=0, only_signal=True)
            train_data = PartNetGraspDataset(zip_file, set_type=0)
            val_data = PartNetGraspDataset(new_dataset_path, set_type=1)  # Use corrected partnet_grasp to validate
            test_data = PartNetGraspDataset(new_dataset_path, set_type=2)  # Use corrected partnet_grasp to test

            _, hist = train_single_imcnn(
                None,
                n_epochs=epochs,
                logging_dir=f"{logging_dir}/imcnn_OldNew_{idx}_trial_{trial_idx}",
                adapt_data=adaptation_data,
                train_data=train_data,
                val_data=val_data,
                test_data=test_data
            )

            test_accuracies[idx].append(hist["test_accuracy"][-1])
            test_losses[idx].append(hist["test_loss"][-1])

    # Compute and store Mann-Whitney U test
    mwutest_file_name = f"{logging_dir}/mann_whitney_u_test.txt"
    if not os.path.exists(mwutest_file_name):
        os.mknod(mwutest_file_name)

    acc_test_statistic, acc_p_value = sp.stats.ranksums(x=test_accuracies[0], y=test_accuracies[1])
    loss_test_statistic, loss_p_value = sp.stats.ranksums(x=test_losses[0], y=test_losses[1])

    with open(mwutest_file_name, "w") as f:
        f.write("### WILCOXON RANK-SUM TEST ###\n")
        f.write(f"Test accuracy statistic: {acc_test_statistic}\n")
        f.write(f"Test accuracy p-value: {acc_p_value}\n")
        f.write(f"Test loss statistic: {loss_test_statistic}\n")
        f.write(f"Test loss p-value: {loss_p_value}\n")
        f.write("###########################\n")

        # Uncorrected test accuracies
        f.write(f"Captured test accuracies (uncorrected): {test_accuracies[0]}\n")
        uncorrected_acc = np.array(test_accuracies[0])
        f.write(
            f"Captured test accuracies (uncorrected) Mean/Std: {uncorrected_acc.mean()} +- {uncorrected_acc.std()}\n"
        )

        # Corrected test accuracies
        f.write(f"Captured test accuracies (deepview-corrected): {test_accuracies[1]}\n")
        corrected_acc = np.array(test_accuracies[1])
        f.write(
            f"Captured test accuracies (deepview-corrected) Mean/Std: {corrected_acc.mean()} +- {corrected_acc.std()}\n"
        )

        f.write("###########################\n")

        # Uncorrected test losses
        f.write(f"Captured test loss (uncorrected): {test_losses[0]}\n")
        uncorrected_loss = np.array(test_losses[0])
        f.write(
            f"Captured test loss (uncorrected) Mean/Std: {uncorrected_loss.mean()} +- {uncorrected_loss.std()}\n"
        )

        # Corrected test losses
        f.write(f"Captured test loss (deepview-corrected): {test_losses[1]}\n")
        corrected_loss = np.array(test_losses[1])
        f.write(
            f"Captured test loss (deepview-corrected) Mean/Std: {corrected_loss.mean()} +- {corrected_loss.std()}\n"
        )

        f.write("###########################\n")



def run_graph(paths,
                        logging_dir,
                        trials=30,
                        epochs=10):
    """Creates the datasets and starts the training runs.

    Parameters
    ----------
    old_dataset_path: str
        The path to the originally pre-processed FAUST dataset
    csv_path: str
        The path to the CSV-file that contains the DeepView-corrections
    new_dataset_path: str
        The path to the DeepView-corrected dataset
    logging_dir: str
        The path to the logging directory
    trials: int
        The amount of trials (i.e. amount of training runs for each corrected and uncorrected sub-experiment)
    epochs: int
        The amount of epochs per trial
    """
    # Create logging dir
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)


    human_corrected_path = "/home/iroberts/projects/VisMeshSegmentation/improve_mesh_segmentation/datasets/partnet_grasp_corrected.zip"
    # Start training runs
    test_accuracies, test_losses  = {},{}
    average_accuracies = []
    average_losses = []
    for idx, zip_file in enumerate(paths):
        # print(zip_file)
        print(idx)
        test_accuracies[idx] = []
        test_losses[idx] = []

        # Train, validate and test IMCNN
        for trial_idx in range(trials):
            print(trial_idx)
            adaptation_data = PartNetGraspDataset(zip_file, set_type=0, only_signal=True)
            train_data = PartNetGraspDataset(zip_file, set_type=0)
            val_data = PartNetGraspDataset(human_corrected_path, set_type=1)  # Use corrected partnet_grasp to validate
            test_data = PartNetGraspDataset(human_corrected_path, set_type=2)  # Use corrected partnet_grasp to test

            _, hist = train_single_imcnn(
                None,
                n_epochs=epochs,
                logging_dir=f"{logging_dir}/imcnn_OldNew_{idx}_trial_{trial_idx}",
                adapt_data=adaptation_data,
                train_data=train_data,
                val_data=val_data,
                test_data=test_data
            )

            test_accuracies[idx].append(hist["test_accuracy"][-1])
            test_losses[idx].append(hist["test_loss"][-1])

        avg_score = np.mean(test_accuracies[idx])
        average_accuracies.append(avg_score)

        avg_loss = np.mean(test_losses[idx])
        average_losses.append(avg_loss)


# Plotting
    # Separate data
    x_values = np.arange(0, 110, 10)  # 0 to 90 (inclusive) -> 10 values
    y_values = average_accuracies[:-1]  # all but the last value
    baseline_value = average_accuracies[-1]  # the last value

    plt.figure(figsize=(8, 6))
    plt.plot(x_values, y_values, marker='o', label='KNN Correction')

    # Add horizontal baseline
    plt.axhline(y=baseline_value, color='r', linestyle='--', label='Human Correction')

    plt.xlabel('Percentage of Noisy Labels Corrected')
    plt.ylabel('Average Accuracy on Test Set')
    plt.title('Effect of KNN-based Label Correction on Accuracy')
    plt.grid(True)
    plt.legend()
    plt.savefig(logging_dir+"/accuracyinf_fig")

    # Separate data
    x_values = np.arange(0, 110, 10)  # 0 to 90 (inclusive) -> 10 values
    y_values = average_losses[:-1]  # all but the last value
    baseline_value = average_losses[-1]  # the last value

    plt.figure(figsize=(8, 6))
    plt.plot(x_values, y_values, marker='o', label='KNN Correction')

    # Add horizontal baseline
    plt.axhline(y=baseline_value, color='r', linestyle='--', label='Human Correction')

    plt.xlabel('Percentage of Noisy Labels Corrected')
    plt.ylabel('Average Losses on Test Set')
    plt.title('Effect of KNN-based Label Correction on Losses')
    plt.grid(True)
    plt.legend()
    plt.savefig(logging_dir+"/lossinf_fig")

