from improve_mesh_segmentation.partnet_grasp.dataset import processed_partnet_grasp_generator

import torch
from torch.utils.data import Dataset
import pandas as pd

import numpy as np
import os


PARTNET_LEN = 100
PARTNET_SPLITS = {
    0: list(range(70)),  # train
    1: list(range(70, 80)),  # validation
    2: list(range(80, PARTNET_LEN)),  # test
    3: list(range(PARTNET_LEN))  # all
}

class PartNetGraspDataset(Dataset):
    """A Pytorch-wrapper class for the PartNet-Grasp dataset."""
    def __init__(
        self,
        path_to_zip,
        correction_file_path,
        set_type=0,
        for_adapt=False,
        set_indices=None,
        device=None
    ):
        self.for_adapt = for_adapt
        self.path_to_zip = path_to_zip
        self.set_type = set_type
        self.device = device
        self.set_indices = set_indices

        # csv = pd.read_csv(correction_file_path, header=None).to_numpy()
        # csv = csv[:, [0, 1, 2]].astype(np.int32)
        assert correction_file_path.name.endswith('.npy'), "Correction file must be a numpy array file (.npy)"
        all_corrs = np.load(correction_file_path)
        for i in range(all_corrs.shape[0]):
            curr_corrs = all_corrs[all_corrs[:, 0] == i]
            assert curr_corrs.shape[0] == curr_corrs[np.unique(curr_corrs[:, 1], return_index=True)[1]].shape[0], "Correction file must contain unique corrections!"

        dataset = processed_partnet_grasp_generator(
            self.path_to_zip,
            set_type=self.set_type,
            set_indices=self.set_indices,
            device=self.device,
        )

        self.point_sets = []
        self.labels = []
        self.true_labels = []
        self.bc = []

        self.shape_indices = []

        ids = PARTNET_SPLITS[self.set_type] if self.set_indices is None else self.set_indices

        curr_idx = 0
        
        for idx, ((point_set, bc), label) in zip(ids, dataset):
            num_verts = point_set.shape[0]

            # get corrections
            corrections = all_corrs[all_corrs[:, 0] == idx]

            true_labels = label.cpu().numpy().copy()
            true_labels[corrections[:, 1]] = 1 - true_labels[corrections[:, 1]] # flips 0 to 1 and vice versa
            self.true_labels.append(true_labels)

            self.point_sets.append(point_set)
            self.labels.append(label)
            self.bc.append(bc)

            self.shape_indices.append(np.arange(curr_idx, curr_idx + num_verts))
            curr_idx += num_verts
        
        self.point_sets = torch.concatenate(self.point_sets, dim=0)
        self.labels = torch.concatenate(self.labels, dim=0)
        self.bc = torch.concatenate(self.bc, dim=0)

        self.true_labels = np.concatenate(self.true_labels)

    def __getitem__(self, idx):
        vert_idxs = self.shape_indices[idx]
        selected_point_set = self.point_sets[vert_idxs]
        selected_label = self.labels[vert_idxs]
        selected_bc = self.bc[vert_idxs]
        selected_true_label = self.true_labels[vert_idxs]
        if self.for_adapt:
            return selected_point_set
        else:
            return (selected_point_set, selected_bc), selected_label, selected_true_label

    def __len__(self):
        return len(self.shape_indices)

    @property
    def num_verts(self):
        return np.sum([len(s) for s in self.shape_indices])

    def ignore_noise_data(self, noisy_data_indices):
        for i, idxs in enumerate(self.shape_indices):
            mask = ~np.isin(idxs, noisy_data_indices)
            self.shape_indices[i] = idxs[mask]

    def get_clean_and_noisy_local_indices(self, noisy_data_indices):
        # Get local indices that should be ignored based on the noisy data indices.
        # i.e. global index n corresponding to local index m in shape_indices[i] should be ignored
        ignore_local_indices = []
        clean_local_indices = []
        for idxs in self.shape_indices:
            mask = np.isin(idxs, noisy_data_indices)
            ignore_local_indices.append(np.where(mask)[0])
            clean_local_indices.append(np.where(~mask)[0])
        return clean_local_indices, ignore_local_indices

    def get_results_from_noisy_data_indices(self, noisy_data_indices):
        content_dict = {"shape_idx": [], "vert_idx": [], "new_label": []}
        for i, idxs in enumerate(self.shape_indices):
            mask = np.isin(idxs, noisy_data_indices)
            selected_idxs = idxs[mask]
            selected_labels = 1 - self.labels[selected_idxs]

            local_idxs = np.where(mask)[0]

            content_dict["shape_idx"].extend([i] * len(selected_idxs))
            content_dict["vert_idx"].extend(local_idxs.tolist())
            content_dict["new_label"].extend(selected_labels.tolist())
        return pd.DataFrame(content_dict)

    def reset(self):
        # noop for compatibility
        pass

class PartNetGraspWithFilterDataset(Dataset):
    """A Pytorch-wrapper class for the PartNet-Grasp dataset."""
    def __init__(
        self,
        path_to_zip,
        correction_file_path,
        set_type=0,
        for_adapt=False,
        set_indices=None,
        device=None
    ):
        self.for_adapt = for_adapt
        self.path_to_zip = path_to_zip
        self.set_type = set_type
        self.device = device
        self.set_indices = set_indices

        csv = pd.read_csv(correction_file_path, header=None).to_numpy()
        csv = csv[:, [0, 1, 2]].astype(np.int32)

        dataset = processed_partnet_grasp_generator(
            self.path_to_zip,
            set_type=self.set_type,
            set_indices=self.set_indices,
            device=self.device,
        )

        self.point_sets = []
        self.labels = []
        self.filter_masks = []
        self.bc = []

        self.shape_indices = []

        ids = PARTNET_SPLITS[self.set_type] if self.set_indices is None else self.set_indices

        curr_idx = 0
        
        for idx, ((point_set, bc), label) in zip(ids, dataset):
            num_verts = point_set.shape[0]
            # get corrections
            corrections = csv[csv[:, 0] == idx]
            corrections = corrections[np.unique(corrections[:, 1], return_index=True)[1]]  # use last update
            use_mask = np.ones_like(label, dtype=bool)
            use_mask[corrections[:, 1]] = False
            self.filter_masks.append(use_mask)

            self.point_sets.append(point_set)
            self.labels.append(label)
            self.bc.append(bc)

            self.shape_indices.append(np.arange(curr_idx, curr_idx + num_verts))
            curr_idx += num_verts
        
        self.point_sets = torch.concatenate(self.point_sets, dim=0)
        self.labels = torch.concatenate(self.labels, dim=0)
        self.bc = torch.concatenate(self.bc, dim=0)

    def __getitem__(self, idx):
        filter_mask = self.filter_masks[idx]
        vert_idxs = self.shape_indices[idx]
        selected_point_set = self.point_sets[vert_idxs]
        selected_label = self.labels[vert_idxs]
        selected_bc = self.bc[vert_idxs]
        if self.for_adapt:
            return selected_point_set
        else:
            return (selected_point_set, selected_bc), selected_label, filter_mask

    def __len__(self):
        return len(self.shape_indices)

    @property
    def num_verts(self):
        return np.sum([len(s) for s in self.shape_indices])

    def reset(self):
        # noop for compatibility
        pass
