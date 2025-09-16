from geoconv_examples.mpi_faust.data.preprocess_faust import get_file_number

from torch.utils.data import IterableDataset
from tqdm import tqdm

import torch
import numpy as np
import os
import random


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


def faust_segmentation_generator(path_to_zip,
                                 set_type=0,
                                 only_signal=False,
                                 device=None,
                                 set_indices=None,
                                 segmentation_labels=None):
    """Reads one element of preprocessed FAUST-geoconv_examples into memory per 'next'-call.

    Parameters
    ----------
    path_to_zip: str
        The path to the .zip-file that contains the preprocessed faust data set
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
    set_indices: list
        A list of integer values that determine which meshes shall be returned. If it is set to 'None', the set
        type determine which meshes will be returned. Defaults to 'None'. Adds noise to barycentric coordinates
        if set type is set to 0.
    segmentation_labels: np.ndarray
        The segmentation labels as a numpy array. If given, the segmentation labels will be returned.

    Returns
    -------
    generator:
        A generator yielding the preprocessed data. I.e. the signal defined on the vertices, the barycentric coordinates
        and the ground truth correspondences.
    """
    # Initialize and sort file names
    dataset = np.load(path_to_zip, allow_pickle=True)
    file_names = [os.path.basename(fn) for fn in dataset.files]
    SIGNAL = [file_name for file_name in file_names if file_name.startswith("SIGNAL")]
    BC = [file_name for file_name in file_names if file_name.startswith("BC")]
    GT = [file_name for file_name in file_names if file_name.startswith("GT")]
    SIGNAL.sort(key=get_file_number), BC.sort(key=get_file_number), GT.sort(key=get_file_number)

    # Set iteration indices according to set type
    if set_indices is None:
        if set_type == 0:
            indices = list(range(70))
            random.shuffle(indices)
        elif set_type == 1:
            indices = range(70, 80)
        elif set_type == 2:
            indices = range(80, 100)
        elif set_type == 3:
            indices = range(100)
        else:
            raise RuntimeError(
                f"There is no 'set_type'={set_type}. Choose from: [0: 'train', 1: 'val', 2: 'test', 3: 'all']."
            )
    else:
        indices = set_indices

    for idx in indices:
        # Read signal
        signal = torch.tensor(dataset[SIGNAL[idx]], dtype=torch.float32)

        # Read bc + add noise
        bc = torch.tensor(dataset[BC[idx]], dtype=torch.float32)

        # Ground truth: Return the indices of the ones for each row
        if segmentation_labels is not None:
            gt = torch.tensor(segmentation_labels[idx], dtype=torch.int64).view(-1,)
        else:
            gt = torch.tensor(dataset[GT[idx]], dtype=torch.int64).view(-1,)

        if device:
            if only_signal:
                yield signal.to(device)
            else:
                yield (signal.to(device), bc.to(device)), gt.to(device)
        else:
            if only_signal:
                yield signal
            else:
                yield (signal, bc), gt


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
        if only_signal:
            self.segmentation_labels = None
        elif os.path.isfile(f"{logging_dir}/noisy_segmentation_labels.npy"):
            self.segmentation_labels = np.load(f"{logging_dir}/noisy_segmentation_labels.npy")
        else:
            self.segmentation_labels = self.get_segmentation_labels()

        # Init dataset
        self.dataset = faust_segmentation_generator(
            self.path_to_zip,
            set_type=self.set_type,
            only_signal=self.only_signal,
            device=self.device,
            segmentation_labels=self.segmentation_labels
        )

    def __iter__(self):
        return self.dataset

    def get_segmentation_labels(self):
        # Load original labels
        segmentation_labels = np.tile(np.load(self.path_to_segmentation_labels), (100, 1))

        # Apply noise to training set only
        noisy_segmentation_labels = []
        dataset = faust_segmentation_generator(
            self.path_to_zip,
            set_type=3,
            only_signal=self.only_signal,
            device=self.device,
            segmentation_labels=segmentation_labels
        )
        for _, labels in tqdm(dataset, desc="Applying symmetric noise to segmentation labels..."):
            noisy_segmentation_labels.append(
                apply_symmetric_noise(segmentation_labels[0][labels.cpu()], noise_level=self.noise_level)
            )
        noisy_segmentation_labels = np.array(noisy_segmentation_labels)

        # Store noisy labels
        os.makedirs(self.logging_dir, exist_ok=True)
        np.save(f"{self.logging_dir}/noisy_segmentation_labels.npy", noisy_segmentation_labels)
        return segmentation_labels

    def reset(self):
        self.dataset = faust_segmentation_generator(
            self.path_to_zip,
            set_type=self.set_type,
            only_signal=self.only_signal,
            device=self.device,
            segmentation_labels=self.segmentation_labels
        )
