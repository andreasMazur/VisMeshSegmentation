import torch
import torch.nn.functional as F
from dataset import PartNetGraspDataset
from models.imcnn import SegImcnn
from models.pointnet import PointNetDense
from pathlib import Path
import numpy as np

from rich.progress import Progress, BarColumn, TimeRemainingColumn, TextColumn, SpinnerColumn, MofNCompleteColumn
from rich.live import Live
from rich.table import Table
from rich.panel import Panel

from knn_utils import calc_knn_graph, calc_topo_weights_with_components_idx
from scipy.stats import mode

from IPython import embed

from termcolor import cprint

def generate_corrections(
    full_dataset: PartNetGraspDataset,
    adapt_dataset: PartNetGraspDataset,
    model_type, # 'imcnn' or 'pointnet'
    start_clean,
    every,
    k_cc,
    k_outlier,
    zeta,
    seed = 42,
    progress=None,
):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cprint(f"Using device: {device}", "blue")

    model_path = Path('fixed_clearing/trained_models', f'{model_type}_{start_clean}_{every}_{k_outlier}_{k_cc}_{zeta}.pth')

    if model_type == 'pointnet':
        model = PointNetDense(k=2).to(device)
        feat_size = 128
    else:
        model = SegImcnn(adapt_data=adapt_dataset).to(device)
        feat_size = 96
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    full_dataset.num_verts

    features_all = np.zeros((full_dataset.num_verts, feat_size))
    labels_all = np.zeros((full_dataset.num_verts, 2))
    train_gt_labels = np.zeros((full_dataset.num_verts,), dtype=np.int64)
    unused_signature_inpt = np.zeros((full_dataset.num_verts,), dtype=np.int64)
    true_labels_all = np.zeros((full_dataset.num_verts,), dtype=np.int64)

    big_comp = set()

    curr_idx = 0
    for step, ((data, bc), labels, true_labels) in enumerate(full_dataset):
        data, bc, labels = data.to(device), bc.to(device), labels.to(device)
        if model_type == 'pointnet':
            probs, feats = model(torch.unsqueeze(data.transpose(0,1), dim=0))
        else:
            probs, feats = model([data, bc])

        probs = probs.squeeze(dim=0)
        feats = feats.squeeze(dim=0)

        features_all[curr_idx:curr_idx+feats.size(0)] = feats.detach().cpu().numpy()
        # prob_all[curr_idx:probs.size(0)] = probs

        labels_all[curr_idx:curr_idx+labels.size(0)] = F.one_hot(labels, num_classes=2).detach().cpu().numpy()
        train_gt_labels[curr_idx:curr_idx+labels.size(0)] = labels.cpu().numpy().astype(np.int64)
        unused_signature_inpt[curr_idx:curr_idx+probs.size(0)] = labels.cpu().numpy().astype(np.int64)
        true_labels_all[curr_idx:curr_idx+true_labels.size] = true_labels

        curr_idx += probs.size(0)

    cprint("\n>> Computing Big components <<", "white")

    train_gt_labels = train_gt_labels.tolist()
    unused_signature_inpt = np.squeeze(unused_signature_inpt).ravel().tolist()

    cprint(f"Calculating topological weights ...", "blue")
    _, idx_of_comp_idx2 = calc_topo_weights_with_components_idx(
        full_dataset.num_verts,
        labels_all,
        torch.from_numpy(features_all).to(device),
        train_gt_labels,
        unused_signature_inpt,
        k=k_cc,
        use_log=False,
        cp_opt=3,
        nclass=2,
        progress=progress,
    )

    curr_big_comp = list(set(range(full_dataset.num_verts)) - set(idx_of_comp_idx2))
    big_comp = big_comp.union(set(curr_big_comp))

    big_comp_idx = list(big_comp)
    feats_big_comp = torch.from_numpy(features_all[big_comp_idx]).to(device)
    labels_big_comp = np.array(train_gt_labels)[big_comp_idx]

    cprint(f">> Calculating kNN graph ...", "blue")
    knnG_list = calc_knn_graph(feats_big_comp, k=k_outlier, refer_trunk_size=5000, query_trunk_size=1000, progress=progress)

    knnG_list = np.array(knnG_list)
    knnG_shape = knnG_list.shape
    knn_labels = labels_big_comp[knnG_list.ravel()]
    knn_labels = np.reshape(knn_labels, knnG_shape)

    cprint(">> Calculating majority vote ...", "blue")
    majority, counts = mode(knn_labels, axis=-1)
    majority = majority.ravel()

    if zeta > 1.0: # use majority vote
        non_outlier_idx = np.where(majority == labels_big_comp)[0]
        outlier_idx = np.where(majority != labels_big_comp)[0]
        cprint(f">> majority == labels_big_comp -> size: {len(non_outlier_idx)}", "white")
    else:  # use zeta filtering
        non_outlier_idx = np.where((majority == labels_big_comp) & (counts >= k_outlier * zeta))[0]
        print(f">> zeta {zeta}, then non_outlier_idx -> size: {len(non_outlier_idx)}")

        outlier_idx = np.where(majority != labels_big_comp)[0]
        outlier_idx = np.array(list(big_comp))[outlier_idx]

    cprint(f">> The number of outliers: {len(outlier_idx)}", "red")
    cprint(f">> The purity of outliers: {np.sum(np.array(train_gt_labels)[outlier_idx] == true_labels_all[outlier_idx])/ float(len(outlier_idx))}", "red")

    big_comp = np.array(list(big_comp))[non_outlier_idx]
    big_comp = set(big_comp.tolist())

    # --- Construct updated dataset with clean data ---
    noisy_data_indices = list(set(range(full_dataset.num_verts)) - big_comp)

    clean_data_num = len(big_comp.intersection(set(np.where(true_labels_all == np.argmax(labels_all, axis=1))[0].tolist())))
    noise_data_num = len(big_comp) - clean_data_num
    cprint(f">> The number of big components: {len(big_comp)}", "yellow")
    cprint(f">> Noise data num: {noise_data_num}", "yellow")
    cprint(f">> Clean data num: {clean_data_num}", "yellow")

    # Compute purity of the component
    cc_size = len(big_comp)
    equal = np.sum(np.argmax(labels_all, axis=1)[list(big_comp)] == true_labels_all[list(big_comp)])
    ratio = equal / float(cc_size)
    cprint(f">> In train clean labels ratio: {ratio:.4f}", "yellow")

    noise_size = len(noisy_data_indices)
    equal = np.sum(np.argmax(labels_all, axis=1)[noisy_data_indices] == true_labels_all[noisy_data_indices])
    cprint(f">> Outside train clean labels ratio: {equal / float(noise_size):.4f}", "yellow")

    df = full_dataset.get_results_from_noisy_data_indices(noisy_data_indices)
    df.to_csv(f'fixed_clearing/final_results/corrections_{model_type}_{start_clean}_{every}_{k_outlier}_{k_cc}_{zeta}.csv', index=False, header=False)

    # embed()

    # curr_idx = 0
    # for shape_idx, local_vert_idx, label in unique_gt_triplets:
    #     global_shape_indices = full_dataset.shape_indices[shape_idx]
    #     local_vert_indices = global_shape_indices - curr_idx
    #     # full_dataset.true_labels
    #     curr_idx += len(global_shape_indices)
        
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate corrections for PartNet-Grasp dataset using trained models.")
    parser.add_argument('--model_type', type=str, choices=['imcnn', 'pointnet', 'all'], default='all', help='Type of model to use for generating corrections.')
    parser.add_argument('--reverse', action='store_true', help='If set, will process models in reverse order.')
    args = parser.parse_args()

    model_dir = Path('fixed_clearing/trained_models')
    models = list(np.sort(list(model_dir.glob('*.pth'))))

    if args.model_type != 'all':
        models = [model for model in models if model.stem.startswith(args.model_type)]
    if args.reverse:
        models = models[::-1]

    # Load dataset
    cprint(">> Loading adapt dataset ...", "blue")
    adapt_dataset = PartNetGraspDataset(
        '../../SegLabelCorrection/data/partnet_grasp/partnet_grasp.zip',
        correction_file_path=Path('unq_dv_triplets.npy'),
        set_type=0,  # Use train data for adaptation
        for_adapt=True,
    )

    cprint(">> Loading full dataset ...", "blue")
    full_dataset = PartNetGraspDataset(
        '../../SegLabelCorrection/data/partnet_grasp/partnet_grasp.zip',
        correction_file_path=Path('unq_dv_triplets.npy'),
        set_type=3,  # Use all data for full processing
        for_adapt=False,
    )

    results_path = Path('fixed_clearing/final_results')

    knn_progress = Progress()
    knn_panel = Panel.fit(
        knn_progress,
        title="Instance progress",
        border_style="cyan",
        padding=(1,2)
    )
    model_proc_progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        MofNCompleteColumn(),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn(),
    )
    progress_table = Table.grid()
    progress_table.add_row(
        Panel.fit(
            model_proc_progress,
            title="Model processing progress",
            border_style="red",
            padding=(1,2)
        ),
        knn_panel,    
    )

    models_to_process = []
    model_task = model_proc_progress.add_task("[blue]Processing models...", total=len(models))
    with Live(progress_table, refresh_per_second=1):
        for model_path in models:
            if model_path.stem.endswith('_tmp'):
                cprint(f"Skipping temporary model file: {model_path.name}", "yellow")
                model_proc_progress.advance(model_task)
            else:
                # Extract parameters from the model filename
                params = model_path.stem.split('_')
                model_type = params[0]
                start_clean = int(params[1])
                every = int(params[2])
                k_outlier = int(params[3])
                k_cc = int(params[4])
                zeta = float(params[5])
                # Check if the model has already been processed
                result_filename = results_path / f'corrections_{model_type}_{start_clean}_{every}_{k_outlier}_{k_cc}_{zeta}.csv'
                if result_filename.is_file():
                    cprint(f"Skipping already processed model: {model_path.name}", "yellow")
                    model_proc_progress.advance(model_task)
                    continue
                else:
                    models_to_process.append(model_path)

        for model_path in models_to_process:
            for task in knn_progress.task_ids:
                knn_progress.remove_task(task)

            # Extract parameters from the model filename
            params = model_path.stem.split('_')
            model_type = params[0]
            start_clean = int(params[1])
            every = int(params[2])
            k_outlier = int(params[3])
            k_cc = int(params[4])
            zeta = float(params[5])
            cprint(f">> Processing model: {model_type} {start_clean} {every} {k_outlier} {k_cc} {zeta}", "green")
            # Check if the model has already been processed
            result_filename = results_path / f'corrections_{model_type}_{start_clean}_{every}_{k_outlier}_{k_cc}_{zeta}.csv'
            if result_filename.is_file():
                cprint(f"Skipping already processed model: {model_path.name}", "yellow")
                model_proc_progress.advance(model_task)
                continue
            
            knn_panel.title = f"Instance progress for {model_path.name}"
            
            # Generate correction
            generate_corrections(
                full_dataset=full_dataset,
                adapt_dataset=adapt_dataset,
                model_type=model_type,
                start_clean=start_clean,
                every=every,
                k_cc=k_cc,
                k_outlier=k_outlier,
                zeta=zeta,
                progress=knn_progress
            )
            print("\n\n")
            model_proc_progress.advance(model_task)
