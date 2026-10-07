import argparse
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from rich.progress import Progress
from scipy.stats import mode
from termcolor import cprint

from geoconv_examples.mpi_faust.data.preprocess_faust import get_file_number

from improve_mesh_segmentation.comparison_methods.topofilter.knn_utils import (
    calc_knn_graph,
    calc_topo_weights_with_components_idx,
)
from improve_mesh_segmentation.comparison_methods.topofilter.models.imcnn import (
    SegImcnn,
)


FAUST_LEN = 100
FAUST_SPLITS = {
    0: list(range(70)),
    1: list(range(70, 80)),
    2: list(range(80, FAUST_LEN)),
    3: list(range(FAUST_LEN)),
}

N_CLASSES = 8
N_FEATURES = 32


def faust_segmentation_generator(
    path_to_zip,
    set_type=0,
    only_signal=False,
    device=None,
    set_indices=None,
    segmentation_labels=None,
    return_noisy_segmentation_labels=None,
    noise_level=None,
    put_into_correct_order=False,
):
    dataset = np.load(path_to_zip, allow_pickle=True)
    file_names = [os.path.basename(fn) for fn in dataset.files]
    signal_files = [file_name for file_name in file_names if file_name.startswith("SIGNAL")]
    bc_files = [file_name for file_name in file_names if file_name.startswith("BC")]
    gt_files = [file_name for file_name in file_names if file_name.startswith("GT")]
    signal_files.sort(key=get_file_number)
    bc_files.sort(key=get_file_number)
    gt_files.sort(key=get_file_number)

    if set_indices is None:
        indices = FAUST_SPLITS[set_type]
    else:
        indices = set_indices

    if isinstance(return_noisy_segmentation_labels, str):
        segmentation_labels = np.load(return_noisy_segmentation_labels)
        assert noise_level is not None, (
            "If noisy segmentation labels are supposed to be returned, the noise level has to be given."
        )

    for idx in indices:
        signal = torch.tensor(dataset[signal_files[idx]], dtype=torch.float32)
        bc = torch.tensor(dataset[bc_files[idx]], dtype=torch.float32)
        gt = torch.tensor(dataset[gt_files[idx]], dtype=torch.int64).view(-1,)

        if segmentation_labels is not None and isinstance(return_noisy_segmentation_labels, str):
            raise RuntimeError("This version is only supposed to load noisy labels from a given path!")
        elif put_into_correct_order:
            gt = segmentation_labels[idx][gt]
        elif segmentation_labels is not None:
            gt = segmentation_labels[idx]
        gt = torch.tensor(gt)

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


class FaustSegmentationDataset:
    def __init__(
        self,
        path_to_zip,
        logging_dir,
        set_type=0,
        for_adapt=False,
        device=None,
        set_indices=None,
        put_into_correct_order=False,
    ):
        self.path_to_zip = path_to_zip
        self.set_type = set_type
        self.device = device
        self.set_indices = set_indices
        self.for_adapt = for_adapt
        self.put_into_correct_order = put_into_correct_order
        self.logging_dir = logging_dir

        if for_adapt:
            self.segmentation_labels = None
        elif os.path.isfile(f"{logging_dir}/noisy_segmentation_labels.npy"):
            print(f"Loading existing noisy segmentation labels from {logging_dir}/noisy_segmentation_labels.npy")
            self.segmentation_labels = np.load(f"{logging_dir}/noisy_segmentation_labels.npy")
        else:
            raise RuntimeError("No noisy segmentation labels found!")

        dataset = faust_segmentation_generator(
            self.path_to_zip,
            set_type=self.set_type,
            only_signal=self.for_adapt,
            device=self.device,
            segmentation_labels=self.segmentation_labels,
            set_indices=self.set_indices,
            put_into_correct_order=False,
        )

        self.point_sets = []
        self.labels = []
        self.bc = []
        self.shape_indices = []

        curr_idx = 0
        for (point_set, bc), label in dataset:
            num_verts = point_set.shape[0]
            self.point_sets.append(point_set)
            self.labels.append(label)
            self.bc.append(bc)
            self.shape_indices.append(np.arange(curr_idx, curr_idx + num_verts))
            curr_idx += num_verts

        self.point_sets = torch.concatenate(self.point_sets, dim=0)
        self.labels = torch.concatenate(self.labels, dim=0)
        self.bc = torch.concatenate(self.bc, dim=0)
        self.random_shape_index_map = np.arange(len(self.shape_indices))

    def set_adapt_mode(self, for_adapt):
        self.for_adapt = for_adapt

    def __getitem__(self, idx):
        non_random_idx = self.random_shape_index_map[idx]
        vert_idxs = self.shape_indices[non_random_idx]
        selected_point_set = self.point_sets[vert_idxs]
        if self.for_adapt:
            return selected_point_set
        selected_label = self.labels[vert_idxs]
        selected_bc = self.bc[vert_idxs]
        return (selected_point_set, selected_bc), selected_label, vert_idxs

    def __len__(self):
        return len(self.shape_indices)

    def shuffle(self):
        random.shuffle(self.random_shape_index_map)

    def unshuffle(self):
        self.random_shape_index_map = np.arange(len(self.shape_indices))

    @property
    def num_verts(self):
        return np.sum([len(s) for s in self.shape_indices])

    def ignore_noise_data(self, noisy_data_indices):
        for i, idxs in enumerate(self.shape_indices):
            mask = ~np.isin(idxs, noisy_data_indices)
            self.shape_indices[i] = idxs[mask]

    def get_results_from_noisy_data_indices(self, noisy_data_indices):
        content_dict = {"shape_idx": [], "point_idx": [], "is_noisy": []}
        for i, idxs in enumerate(self.shape_indices):
            mask = np.isin(idxs, noisy_data_indices)
            selected_idxs = idxs[mask]

            content_dict["shape_idx"].extend([i] * len(selected_idxs))
            content_dict["point_idx"].extend(selected_idxs.tolist())
            content_dict["is_noisy"].extend([True] * len(selected_idxs))
        return pd.DataFrame(content_dict)


def compute_noisy_dataset_indices(ds, full_dataset_length, mdl, device, k_cc, k_outlier, zeta):
    ds.unshuffle()
    big_comp = set()

    with torch.no_grad():
        features_all = np.zeros((full_dataset_length, N_FEATURES))
        labels_all = np.zeros((full_dataset_length, N_CLASSES))
        train_gt_labels = np.zeros((full_dataset_length,), dtype=np.int64)
        unused_signature_inpt = np.zeros((full_dataset_length,), dtype=np.int64)

        for _, ((data, bc), labels, vert_idx) in enumerate(ds):
            data, bc, labels = data.to(device), bc.to(device), labels.to(device)
            probs, feats = mdl([data, bc])

            feats = feats.squeeze(dim=0)
            features_all[vert_idx] = feats.detach().cpu().numpy()
            labels_all[vert_idx] = F.one_hot(labels, num_classes=N_CLASSES).detach().cpu().numpy()
            train_gt_labels[vert_idx] = labels.cpu().numpy().astype(np.int64)
            unused_signature_inpt[vert_idx] = labels.cpu().numpy().astype(np.int64)

        cprint("\n>> Computing Big components <<", "white")

        train_gt_labels = train_gt_labels.tolist()
        unused_signature_inpt = np.squeeze(unused_signature_inpt).ravel().tolist()

        cprint("Calculating topological weights ...", "blue")
        _, idx_of_comp_idx2 = calc_topo_weights_with_components_idx(
            full_dataset_length,
            labels_all,
            torch.from_numpy(features_all).to(device),
            train_gt_labels,
            unused_signature_inpt,
            k=k_cc,
            use_log=False,
            cp_opt=3,
            nclass=N_CLASSES,
        )

        curr_big_comp = list(set(range(full_dataset_length)) - set(idx_of_comp_idx2))
        big_comp = big_comp.union(set(curr_big_comp))

        big_comp_idx = list(big_comp)
        feats_big_comp = torch.from_numpy(features_all[big_comp_idx]).to(device)
        labels_big_comp = np.array(train_gt_labels)[big_comp_idx]

        cprint(">> Calculating kNN graph ...", "blue")
        knnG_list = calc_knn_graph(
            feats_big_comp,
            k=k_outlier,
            refer_trunk_size=5000,
            query_trunk_size=1000,
        )

        knnG_list = np.array(knnG_list)
        knnG_shape = knnG_list.shape
        knn_labels = labels_big_comp[knnG_list.ravel()]
        knn_labels = np.reshape(knn_labels, knnG_shape)

        cprint(">> Calculating majority vote ...", "blue")
        majority, counts = mode(knn_labels, axis=-1)
        majority = majority.ravel()

        if zeta > 1.0:
            non_outlier_idx = np.where(majority == labels_big_comp)[0]
            outlier_idx = np.where(majority != labels_big_comp)[0]
            cprint(f">> majority == labels_big_comp -> size: {len(non_outlier_idx)}", "white")
        else:
            non_outlier_idx = np.where((majority == labels_big_comp) & (counts >= k_outlier * zeta))[0]
            print(f">> zeta {zeta}, then non_outlier_idx -> size: {len(non_outlier_idx)}")
            outlier_idx = np.where(majority != labels_big_comp)[0]
            outlier_idx = np.array(list(big_comp))[outlier_idx]

        cprint(f">> The number of outliers: {len(outlier_idx)}", "red")

        big_comp = np.array(list(big_comp))[non_outlier_idx]
        big_comp = set(big_comp.tolist())

        noisy_data_indices = list(set(range(full_dataset_length)) - big_comp)
        cprint(f">> The number of big components: {len(big_comp)}", "yellow")
        return noisy_data_indices


def train(
    data_path,
    logging_dir: Path,
    n_epochs=90,
    seed=42,
    milestone=(30, 60),
    denoise_every_n_epoch=5,
    when_to_denoise=30,
    k_outlier=32,
    k_cc=4,
    zeta=0.5,
):
    topofilter_logging_dir = logging_dir / "topofilter_logs" / f"{when_to_denoise}_{denoise_every_n_epoch}_{k_outlier}_{k_cc}_{zeta}"
    if (topofilter_logging_dir / "corrections_filter_iter_05.csv").is_file():
        print(f"skipping already computed {when_to_denoise}_{denoise_every_n_epoch}_{k_outlier}_{k_cc}_{zeta}...")
        return
    topofilter_logging_dir.mkdir(exist_ok=True, parents=True)

    lr = 0.001
    gamma = 0.5
    weight_decay = 1e-4

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cprint(f"Using device: {device}", "blue")

    train_dataset = FaustSegmentationDataset(
        path_to_zip=data_path,
        logging_dir=logging_dir.as_posix(),
        set_type=0,
    )
    train_len = len(train_dataset)
    curr_train_set = train_dataset

    full_dataset = FaustSegmentationDataset(
        path_to_zip=data_path,
        logging_dir=logging_dir.as_posix(),
        set_type=3,
    )

    train_dataset.set_adapt_mode(True)
    model = SegImcnn(
        adapt_data=train_dataset,
        signal_dim=544,
        kernel_size=(3, 6),
        segmentation_classes=N_CLASSES,
        template_radius=0.027744965069279016,
        layer_conf=[(32, 6), (32, 6)],
    ).to(device)
    train_dataset.set_adapt_mode(False)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(milestone), gamma=gamma)

    clean_iteration_idx = 0
    full_set_verts = train_dataset.num_verts

    cprint("Starting training...", "green")
    for epoch in range(n_epochs):
        train_dataset.shuffle()
        with Progress() as progress:
            epoch_task = progress.add_task(f"[Epoch {epoch+1}/{n_epochs}] Training...", total=train_len)

            epoch_accuracy = 0.0
            epoch_loss = 0.0
            mean_accuracy = 0.0
            mean_loss = 0.0
            correct_verts = 0
            total_verts = 0

            model.train()
            for step, ((data, bc), labels, _) in enumerate(curr_train_set):
                if len(data) == 0:
                    continue

                data, bc, labels = data.to(device), bc.to(device), labels.to(device)
                optimizer.zero_grad()
                pred = model([data, bc])[0]
                loss = F.cross_entropy(pred, labels, reduction="none")
                loss = loss.mean()
                loss.backward()
                optimizer.step()

                epoch_loss = epoch_loss + loss.detach()
                predicted_classes = torch.argmax(pred, dim=1).detach()
                correct_local_verts = (predicted_classes == labels)
                correct_verts += correct_local_verts.sum().item()

                epoch_accuracy += correct_local_verts.sum() / correct_local_verts.size(0)
                mean_accuracy = epoch_accuracy / (step + 1)
                mean_loss = epoch_loss / (step + 1)
                total_verts += correct_local_verts.size(0)

                progress.update(
                    epoch_task,
                    advance=1,
                    description=(
                        f"[Epoch {epoch+1}/{n_epochs}] Loss: {mean_loss.item():.4f} - "
                        f"Accuracy: {mean_accuracy.item():.4f} - Correct vertices: {correct_verts}/{total_verts}"
                    ),
                )
            progress.stop_task(epoch_task)
            scheduler.step()
            model.eval()

        if epoch >= when_to_denoise and (epoch - when_to_denoise) % denoise_every_n_epoch == 0:
            noisy_data_indices_train = compute_noisy_dataset_indices(
                train_dataset,
                full_set_verts,
                model,
                device,
                k_cc,
                k_outlier,
                zeta,
            )
            train_set_ignore_noisy = FaustSegmentationDataset(
                path_to_zip=data_path,
                logging_dir=logging_dir.as_posix(),
                set_type=0,
            )

            noisy_data_indices = compute_noisy_dataset_indices(
                full_dataset,
                full_dataset.num_verts,
                model,
                device,
                k_cc,
                k_outlier,
                zeta,
            )

            df = full_dataset.get_results_from_noisy_data_indices(noisy_data_indices)
            df.to_csv(
                topofilter_logging_dir / f"corrections_filter_iter_{clean_iteration_idx:02d}.csv",
                header=False,
                index=False,
            )
            torch.save(model.state_dict(), topofilter_logging_dir / f"model_filter_iter_{clean_iteration_idx:02d}.pt")
            clean_iteration_idx += 1

            train_set_ignore_noisy.ignore_noise_data(noisy_data_indices_train)


def run_topofilter_faust_experiments(
    dataset_path,
    logging_dir,
    topofilter_params,
    seed=42,
    milestone=(30, 60),
):
    dataset_path = Path(dataset_path)
    logging_dir = Path(logging_dir)

    for k_cc in topofilter_params["k_ccs"]:
        for k_outlier in topofilter_params["k_outliers"]:
            for every in topofilter_params["everys"]:
                for start_clean in topofilter_params["start_cleans"]:
                    for zeta in topofilter_params["zetas"]:
                        n_epochs = (start_clean + 1) + 5 * every
                        cprint(
                            f"\n\n>> Training with start_clean={start_clean}, every={every}, "
                            f"k_outlier={k_outlier}, k_cc={k_cc}, zeta={zeta} for {n_epochs} epochs <<",
                            "green",
                        )
                        train(
                            data_path=dataset_path.as_posix(),
                            logging_dir=logging_dir,
                            denoise_every_n_epoch=every,
                            when_to_denoise=start_clean,
                            k_outlier=k_outlier,
                            k_cc=k_cc,
                            zeta=zeta,
                            n_epochs=n_epochs,
                            seed=seed,
                            milestone=milestone,
                        )