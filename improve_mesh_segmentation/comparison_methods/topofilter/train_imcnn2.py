from dataset import PartNetGraspDataset
from models.imcnn import SegImcnn

from knn_utils import calc_knn_graph, calc_topo_weights_with_components_idx
from scipy.stats import mode

import torch
import torch.optim as optim
import torch.nn.functional as F
from torcheval.metrics.functional import multiclass_accuracy

import numpy as np
from pathlib import Path

from rich.progress import track, Progress, BarColumn, TimeRemainingColumn, TextColumn, SpinnerColumn, MofNCompleteColumn
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from termcolor import cprint
from IPython import embed

def train(
        data_path,
        correction_file_path,
        knn_progress,
        train_progress,
        seed=42,
        milestone = [30, 60],
        n_epochs=90,
        denoise_every_n_epoch=5,
        when_to_denoise=30,
        k_outlier=32,
        k_cc=4,
        zeta=0.5,
    ):

    lr = 0.001
    gamma = 0.5
    weight_decay = 1e-4

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    cprint(f"Using device: {device}", "blue")

    train_dataset = PartNetGraspDataset(
        path_to_zip=data_path,
        correction_file_path=correction_file_path,
        set_type=0,
    )
    train_len = len(train_dataset)
    curr_train_set = train_dataset

    val_dataset = PartNetGraspDataset(
        path_to_zip=data_path,
        correction_file_path=correction_file_path,
        set_type=1,
    )
    val_len = len(val_dataset)

    model = SegImcnn(adapt_data=PartNetGraspDataset(path_to_zip=data_path, correction_file_path=correction_file_path, for_adapt=True)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestone, gamma=gamma)

    best_acc = 0.
    best_epoch = 0
    best_weights = None

    big_comp = set()
    patience = 65
    no_improve_counter = 0

    best_num_clean_labels = 0
    best_noisy_data_indices = []
    
    full_set_verts = train_dataset.num_verts

    stats = {
        'train_set_size': [],
        'clean_in_set': [],
        'noisy_in_set': [],
        'noisy_outside_train': [],
        'train_acc': [],
        'train_loss': [],
        'val_acc': [],
        'val_loss': [],
        'clean_outside_train': [],
    }

    cprint("Starting training...", "green")
    clean_data_local_indices = [np.empty(0) for _ in range(train_len)]
    for epoch in range(n_epochs):
        epoch_task = train_progress.add_task(f"[Epoch {epoch+1}/{n_epochs}] Training...", total=train_len)

        val_total = 0
        val_correct = 0
        val_loss = 0.
        val_accuracy = 0.
        mean_val_accuracy = 0.
        mean_val_loss = 0.

        epoch_accuracy = 0.
        epoch_loss = 0.
        mean_accuracy = 0.
        mean_loss = 0.
        correct_verts = 0
        total_verts = 0

        model.train()
        for step, ((data, bc), labels, _) in enumerate(curr_train_set):
            shape_index = step
            # Get local noisy vertex indices for the current shape
            local_clean_vertex_indices = clean_data_local_indices[shape_index]

            # In case all vertices are filtered out, skip this shape
            if len(local_clean_vertex_indices) > 0 and len(data[local_clean_vertex_indices]) == 0:
                continue                        

            data, bc, labels = data.to(device), bc.to(device), labels.to(device)
            optimizer.zero_grad()
            pred = model([data, bc])[0]
            loss = F.nll_loss(pred, labels, reduction='none') # get per-vertex loss

            # ignore filtered vertices in loss computation
            if len(local_clean_vertex_indices) > 0:
                loss = loss[local_clean_vertex_indices]  
            # average loss over all remaining vertices
            loss = loss.mean()

            loss.backward()
            optimizer.step()

            epoch_loss = epoch_loss + loss.detach()

            predicted_classes = torch.argmax(pred, dim=1).detach()
            correct_local_verts = (predicted_classes == labels)
            if len(local_clean_vertex_indices) > 0:
                correct_local_verts = correct_local_verts[local_clean_vertex_indices]
            correct_verts += correct_local_verts.sum().item()

            epoch_accuracy += correct_local_verts.sum()/correct_local_verts.size(0)
            
            mean_accuracy = epoch_accuracy / (step + 1)
            mean_loss = epoch_loss / (step + 1)

            total_verts += correct_local_verts.size(0)

            train_progress.update(epoch_task, advance=1, description=f"[Epoch {epoch+1}/{n_epochs}] Loss: {mean_loss.item():.4f} - Accuracy: {mean_accuracy.item():.4f} - Correct vertices: {correct_verts}/{total_verts}")
        train_progress.stop_task(epoch_task)
        scheduler.step()
        
        # Compute connected components

        model.eval()
        # -- Validation --
        val_task = train_progress.add_task(f"[Epoch {epoch+1}/{n_epochs}] Validation...", total=val_len)
        with torch.no_grad():
            for step, ((data, bc), _, labels) in enumerate(val_dataset):
                data, bc, labels = data.to(device), bc.to(device), torch.from_numpy(labels).to(device)
                pred = model([data, bc])[0]
                loss = F.nll_loss(pred, labels)

                val_loss += loss.item()
                val_accuracy += multiclass_accuracy(pred, labels).item()

                predicted_classes = torch.argmax(pred, dim=1).detach()
                val_correct += (predicted_classes == labels).sum().item()
                val_total += labels.size(0)

                mean_val_accuracy = val_accuracy / (step + 1)
                mean_val_loss = val_loss / (step + 1)

                train_progress.update(val_task, advance=1, description=f"[Epoch {epoch+1}/{n_epochs}] Val Loss: {mean_val_loss:.4f} - Val Accuracy: {mean_val_accuracy:.4f} - Correct vertices: {val_correct}/{val_total}")
            train_progress.remove_task(val_task)
            train_progress.update(epoch_task, description=f"[Epoch {epoch+1}/{n_epochs}] Loss: {mean_loss.item():.4f} - Accuracy: {mean_accuracy.item():.4f} - Correct vertices: {correct_verts}/{total_verts} - Val Loss: {mean_val_loss:.4f} - Val Accuracy: {mean_val_accuracy:.4f}")

        if epoch >= when_to_denoise and (epoch - when_to_denoise) % denoise_every_n_epoch == 0:
            features_all = np.zeros((full_set_verts, 96))
            # prob_all = torch.zeros((full_set_verts, 2)).to(device)
            labels_all = np.zeros((full_set_verts, 2))
            train_gt_labels = np.zeros((full_set_verts,), dtype=np.int64)
            unused_signature_inpt = np.zeros((full_set_verts,), dtype=np.int64) #https://github.com/pxiangwu/TopoFilter/issues/3
            true_labels_all = np.zeros((full_set_verts,), dtype=np.int64)

            curr_idx = 0
            for step, ((data, bc), labels, true_labels) in enumerate(train_dataset):
                data, bc, labels = data.to(device), bc.to(device), labels.to(device)
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
                full_set_verts,
                labels_all,
                torch.from_numpy(features_all).to(device),
                train_gt_labels,
                unused_signature_inpt,
                k=k_cc,
                use_log=False,
                cp_opt=3,
                nclass=2,
                progress=knn_progress,
            )

            curr_big_comp = list(set(range(full_set_verts)) - set(idx_of_comp_idx2))
            big_comp = big_comp.union(set(curr_big_comp))

            big_comp_idx = list(big_comp)
            feats_big_comp = torch.from_numpy(features_all[big_comp_idx]).to(device)
            labels_big_comp = np.array(train_gt_labels)[big_comp_idx]

            cprint(f">> Calculating kNN graph ...", "blue")
            knnG_list = calc_knn_graph(feats_big_comp, k=k_outlier, refer_trunk_size=5000, query_trunk_size=1000, progress=knn_progress)

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
            train_set_ignore_noisy = PartNetGraspDataset(
                path_to_zip=data_path,
                correction_file_path=correction_file_path,
                set_type=0,
            )
            noisy_data_indices = list(set(range(full_set_verts)) - big_comp)

            # Get local noisy vertex indices based on the global indices *before* filtering out!
            clean_data_local_indices, noisy_data_local_indices = train_set_ignore_noisy.get_clean_and_noisy_local_indices(noisy_data_indices)

            # Here we only filter out noisy data through local indices, so we don't need to update the set
            # we still keep it to display the stats 
            train_set_ignore_noisy.ignore_noise_data(noisy_data_indices)
            # curr_train_set = train_set_ignore_noisy

            clean_data_num = len(big_comp.intersection(set(np.where(true_labels_all == np.argmax(labels_all, axis=1))[0].tolist())))
            noise_data_num = len(big_comp) - clean_data_num
            cprint(f">> The number of big components: {len(big_comp)}", "yellow")
            cprint(f">> Noise data num: {noise_data_num}", "yellow")
            cprint(f">> Clean data num: {clean_data_num}", "yellow")

            if clean_data_num > best_num_clean_labels:
                cprint(f">> Found better big component with {clean_data_num} vertices", "green")
                best_num_clean_labels = clean_data_num
                best_noisy_data_indices = noisy_data_indices
                # Save model
                torch.save(model.state_dict(), f'./trained_models/imcnn_{when_to_denoise}_{denoise_every_n_epoch}_{k_outlier}_{k_cc}_{zeta}_tmp.pth')

            # Compute purity of the component
            cc_size = len(big_comp)
            equal = np.sum(np.argmax(labels_all, axis=1)[list(big_comp)] == true_labels_all[list(big_comp)])
            ratio = equal / float(cc_size)
            cprint(f">> In train clean labels ratio: {ratio:.4f}", "yellow")

            noise_size = len(noisy_data_indices)
            equal = np.sum(np.argmax(labels_all, axis=1)[noisy_data_indices] == true_labels_all[noisy_data_indices])
            cprint(f">> Outside train clean labels ratio: {equal / float(noise_size):.4f}", "yellow")

            ### Collect stats
            stats['train_set_size'].append(len(big_comp))
            stats['clean_in_set'].append(clean_data_num)
            stats['noisy_in_set'].append(noise_data_num)
            stats['train_acc'].append(mean_accuracy.item())
            stats['train_loss'].append(mean_loss.item())
            stats['noisy_outside_train'].append(np.sum(np.argmax(labels_all, axis=1)[noisy_data_indices] != true_labels_all[noisy_data_indices]))
            stats['clean_outside_train'].append(np.sum(np.argmax(labels_all, axis=1)[noisy_data_indices] == true_labels_all[noisy_data_indices]))
            stats['val_acc'].append(mean_val_accuracy)
            stats['val_loss'].append(mean_val_loss)

            np.save(f"results/imcnn_stats_{when_to_denoise}_{denoise_every_n_epoch}_{k_outlier}_{k_cc}_{zeta}.npy", stats)

    df = train_dataset.get_results_from_noisy_data_indices(best_noisy_data_indices)
    df.to_csv(f'results/imcnn_corrections_{when_to_denoise}_{denoise_every_n_epoch}_{k_outlier}_{k_cc}_{zeta}.csv', header=False, index=False)
    best_model = Path(f'./trained_models/imcnn_{when_to_denoise}_{denoise_every_n_epoch}_{k_outlier}_{k_cc}_{zeta}_tmp.pth')
    best_model.rename(best_model.with_stem(best_model.stem.replace('_tmp', '')))



if __name__ == "__main__":

    every = 10
    start_clean = 15
    k_outlier = 32
    k_cc = 5
    zeta = 0.5

    # train(
    #     data_path="../../SegLabelCorrection/data/partnet_grasp/partnet_grasp.zip",
    #     correction_file_path="../improve_mesh_segmentation/data_correction/partnet_correction.csv",
    #     denoise_every_n_epoch=every,
    #     when_to_denoise=start_clean,
    #     k_outlier=k_outlier,
    #     k_cc=k_cc,
    #     zeta=zeta,
    # )

    # every =       [ 5,  5,  5, 10,  5,  5,  5,  5]
    # start_clean = [ 5,  5, 10, 10, 15, 20, 25, 30]
    # k_outlier =   [32, 32, 32, 32, 32, 32, 32, 32]
    # k_cc =        [ 4,  5,  5,  5,  5,  5,  5,  5]
    # zeta =        [.5, .5, .5, .5, .5, .5, .5, .5] 


    k_ccs = [
        10, 50, 250#, 1000, 3000, 5000
    ]

    start_cleans = [1, 2, 3, 4]

    everys = [1, 2, 3]

    zetas = [
        0.1, 0.25, 0.5, 0.75, 0.9, 1.0
    ]                

    k_outliers = [32, 18, 64]

    total_num_models = len(start_cleans) * len(everys) * len(k_ccs) * len(zetas) * len(k_outliers)

    overall_progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        MofNCompleteColumn(),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn(),
    )
    overall_panel = Panel.fit(
        overall_progress,
        title="Ablation Model Train Progress",
        border_style='red',
        padding=(2,2),
    )

    knn_progress = Progress()
    knn_panel = Panel.fit(
        knn_progress,
        title="Filter progress",
        border_style='cyan',
        padding=(1,1),
    )

    train_progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn(),
    )
    train_panel = Panel.fit(
        train_progress,
        title="Train Progress",
        border_style='green',
        padding=(1,1),
    )

    progress_table = Table.grid()
    progress_table.add_row(overall_panel)
    progress_table.add_row(
        train_panel,
        knn_panel,
    )
    
    with Live(progress_table):
        overall_task = overall_progress.add_task("[blue]Training Models...", total=total_num_models)
        for k_cc in k_ccs:
            for k_outlier in k_outliers:
                for every in everys:
                    for start_clean in start_cleans:
                        for zeta in zetas:
                            for task in knn_progress.task_ids:
                                knn_progress.remove_task(task)
                            for task in train_progress.task_ids:
                                train_progress.remove_task(task)
                            knn_panel.title = f'Filtering progress for imcnn_{start_clean}_{every}_{k_outlier}_{k_cc}_{zeta}'
                            train_panel.title = f'Train progress for imcnn_{start_clean}_{every}_{k_outlier}_{k_cc}_{zeta}'

                            n_epochs = (start_clean+1)+5*every
                            cprint(f"\n\n>> Training with start_clean={start_clean}, every={every}, k_outlier={k_outlier}, k_cc={k_cc}, zeta={zeta} for {n_epochs} epochs <<", "green")
                            csv_path = Path(f'./results/imcnn_corrections_{start_clean}_{every}_{k_outlier}_{k_cc}_{zeta}.csv')
                            if csv_path.exists():
                                cprint(f"Skipping {csv_path} as it already exists", "yellow")
                                overall_progress.advance(overall_task)
                                continue
                            train(
                                data_path="../../SegLabelCorrection/data/partnet_grasp/partnet_grasp.zip",
                                correction_file_path="../improve_mesh_segmentation/data_correction/partnet_correction.csv",
                                denoise_every_n_epoch=every,
                                when_to_denoise=start_clean,
                                k_outlier=k_outlier,
                                k_cc=k_cc,
                                zeta=zeta,
                                n_epochs=n_epochs,
                                knn_progress=knn_progress,
                                train_progress=train_progress,
                            )
                            overall_progress.advance(overall_task)
