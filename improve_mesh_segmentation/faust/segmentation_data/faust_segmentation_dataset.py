from geoconv_examples.mpi_faust.pytorch.faust_data_set import faust_generator

from torch.utils.data import IterableDataset

import torch
import numpy as np
import os


def apply_symmetric_noise(labels, noise_level):
    """Applies symmetric labels noise to a given label matrix

    Parameters
    ----------
    labels: np.ndarray
        The original labels as a numpy array.
    noise_level: float
        The chance of flipping a label to one other class. The probability of keeping the original label is
        1 - 7 * noise_level, since there are 8 classes in total.

    Returns
    -------
    np.ndarray
        The noisy labels as a numpy array.
    """
    noisy_labels = []
    for label in labels:
        maintain_true_class_probability = 1 - 7 * noise_level

        p = np.full(shape=(8,), fill_value=noise_level)
        p[label] = maintain_true_class_probability
        noisy_labels.append(np.random.choice(range(8), p=p))

    return np.array(noisy_labels)


def faust_segmentation_generator(faust_dataset_path,
                                 segmentation_labels,
                                 set_type=0,
                                 only_signal=False,
                                 device=None):
    """Generator that yields FAUST data along with segmentation labels.

    Parameters
    ----------
    faust_dataset_path: str
        Path to the FAUST dataset zip file.
    segmentation_labels: np.ndarray
        The segmentation labels as a numpy array.
    set_type: int
        This integer has to be either:
            - 0 -> "train"  (adds noise to barycentric coordinates)
            - 1 -> "validation"
            - 2 -> "test"
            - 3 -> "all meshes"
        Depending on the choice, the training-, validation or testing data set will be returned. The split is equal to
        the one given in:
        > [Geodesic Convolutional Neural Networks on Riemannian Manifolds](https://arxiv.org/abs/1501.06297)
        > Jonathan Masci and Davide Boscaini et al.
    only_signal: bool
        Return only the signal matrices.
    device:
        The device to put the data on.
    """
    dataset = faust_generator(
        path_to_zip=faust_dataset_path,
        set_type=set_type,
        only_signal=only_signal,
        device=device,
        return_coordinates=False
    )

    if only_signal:
        for signal in dataset:
            yield signal
    else:
        for (shot, bc), labels in dataset:
            yield (shot, bc), torch.tensor(segmentation_labels[labels.cpu()]).to(device)


class FaustSegmentationDataset(IterableDataset):
    def __init__(self,
                 path_to_zip,
                 path_to_segmentation_labels,
                 logging_dir,
                 set_type=0,
                 only_signal=False,
                 device=None,
                 noise_level=0.0):
        self.path_to_zip = path_to_zip
        self.path_to_segmentation_labels = path_to_segmentation_labels
        self.set_type = set_type
        self.only_signal = only_signal
        self.device = device

        # Remember noisy labels
        self.logging_dir = logging_dir
        self.noise_level = noise_level
        if os.path.isfile(f"{logging_dir}/noisy_segmentation_labels.npy"):
            self.segmentation_labels = np.load(f"{logging_dir}/noisy_segmentation_labels.npy")
        else:
            self.segmentation_labels = self.get_segmentation_labels()

        # Init dataset
        self.dataset = faust_segmentation_generator(
            self.path_to_zip,
            self.segmentation_labels,
            set_type=self.set_type,
            only_signal=self.only_signal,
            device=self.device
        )

    def __iter__(self):
        return self.dataset

    def get_segmentation_labels(self):
        segmentation_labels = np.load(self.path_to_segmentation_labels)
        segmentation_labels = apply_symmetric_noise(segmentation_labels, noise_level=self.noise_level)
        os.makedirs(self.logging_dir, exist_ok=True)
        np.save(f"{self.logging_dir}/noisy_segmentation_labels.npy", segmentation_labels)
        return segmentation_labels

    def reset(self):
        self.dataset = faust_segmentation_generator(
            self.path_to_zip,
            self.segmentation_labels,
            set_type=self.set_type,
            only_signal=self.only_signal,
            device=self.device
        )
