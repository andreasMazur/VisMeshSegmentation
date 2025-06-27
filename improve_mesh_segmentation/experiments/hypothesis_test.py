from improve_mesh_segmentation.partnet_grasp.dataset import PartNetGraspDataset
from improve_mesh_segmentation.data_correction.convert_partnet import convert_partnet
from improve_mesh_segmentation.training.train_imcnn import train_single_imcnn

from pathlib import Path

import os
import scipy as sp
import numpy as np
import torch


def run_hypothesis_test(old_dataset_path,
                        new_dataset_path,
                        csv_path,
                        logging_dir,
                        trials=30,
                        epochs=10,
                        clean_data_path=None):
    """Creates the datasets and starts the training runs.

    Parameters
    ----------
    old_dataset_path: str
        The path to the originally pre-processed FAUST dataset
    new_dataset_path: str
        The path to the DeepView-corrected dataset
    csv_path: str
        The path to the CSV-file that contains the DeepView-corrections
    logging_dir: str
        The path to the logging directory
    trials: int
        The amount of trials (i.e. amount of training runs for each corrected and uncorrected sub-experiment)
    epochs: int
        The amount of epochs per trial
    clean_data_path: str | None
        The path to a dataset that is considered "clean" and can be used to evaluate test performance.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create logging dir
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)

    # Integrate DeepView-corrected labels into the dataset
    if not Path(f"{new_dataset_path}.zip").is_file():
        convert_partnet(old_data_path=old_dataset_path, new_data_path=new_dataset_path, csv_path=csv_path)
    else:
        print(f"Found converted dataset file: '{new_dataset_path}'. Skipping preprocessing.")
    new_dataset_path = f"{new_dataset_path}.zip"
    old_dataset_path = f"{old_dataset_path}.zip"

    # Start training runs
    test_accuracies, test_losses = [[], []], [[], []]
    for idx, zip_file in enumerate([old_dataset_path, new_dataset_path]):
        if clean_data_path is None:
            clean_data_path = new_dataset_path

        # Train, validate and test IMCNN
        for trial_idx in range(trials):
            print(f"\nUsing un-corrected data: {idx == 0} | Using corrected data: {idx == 1} | Trial {trial_idx}")
            adaptation_data = PartNetGraspDataset(zip_file, set_type=0, only_signal=True, device=device)
            train_data = PartNetGraspDataset(zip_file, set_type=0, device=device)
            val_data = PartNetGraspDataset(clean_data_path, set_type=1, device=device)  # Use human-expert corrected to validate
            test_data = PartNetGraspDataset(clean_data_path, set_type=2, device=device)  # Use human-expert corrected to test

            _, hist = train_single_imcnn(
                None,
                n_epochs=epochs,
                device=device,
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
