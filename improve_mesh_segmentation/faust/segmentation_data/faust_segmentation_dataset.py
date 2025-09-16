from geoconv_examples.mpi_faust.pytorch.faust_data_set import faust_generator

from torch.utils.data import IterableDataset

import torch
import numpy as np


def faust_segmentation_generator(faust_dataset_path,
                                 segmentation_labels_path,
                                 set_type=0,
                                 only_signal=False,
                                 device=None):
    """Generator that yields FAUST data along with segmentation labels.

    Parameters
    ----------
    faust_dataset_path: str
        Path to the FAUST dataset zip file.
    segmentation_labels_path: str
        Path to the numpy file containing segmentation labels.
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
    segmentation_labels = np.load(segmentation_labels_path)

    if only_signal:
        for signal in dataset:
            yield signal
    else:
        for (shot, bc), labels in dataset:
            yield (shot, bc), torch.tensor(segmentation_labels[labels].cpu()).to(device)


class FaustSegmentationDataset(IterableDataset):
    def __init__(self, path_to_zip, path_to_segmentation_labels, set_type=0, only_signal=False, device=None):
        self.path_to_zip = path_to_zip
        self.path_to_segmentation_labels = path_to_segmentation_labels
        self.set_type = set_type
        self.only_signal = only_signal
        self.device = device

        self.dataset = faust_segmentation_generator(
            self.path_to_zip,
            self.path_to_segmentation_labels,
            set_type=self.set_type,
            only_signal=self.only_signal,
            device=self.device
        )

    def __iter__(self):
        return self.dataset

    def reset(self):
        self.dataset = faust_segmentation_generator(
            self.path_to_zip,
            self.path_to_segmentation_labels,
            set_type=self.set_type,
            only_signal=self.only_signal,
            device=self.device
        )
