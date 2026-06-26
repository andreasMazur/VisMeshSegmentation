from dataset import PartNetGraspDataset, PartNetGraspWithFilterDataset
from models.imcnn import SegImcnn

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from rich.progress import Progress
from termcolor import cprint

def train_with_corrected(
    data_path,
    manual_correction_file_path,
    auto_correction_file_path,
    seed=42,
    n_epochs=20,
    remove_progress=False
):
    lr = 0.001
    weight_decay = 1e-4

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = PartNetGraspDataset(
        path_to_zip=data_path,
        correction_file_path=auto_correction_file_path,
        set_type=0,  # train
    )

    validation_dataset = PartNetGraspDataset(
        path_to_zip=data_path,
        correction_file_path=manual_correction_file_path,
        set_type=1,  # validation
    )

    test_dataset = PartNetGraspDataset(
        path_to_zip=data_path,
        correction_file_path=manual_correction_file_path,
        set_type=2,  # test
    )

    model = SegImcnn(
        adapt_data=PartNetGraspDataset(
            path_to_zip=data_path,
            correction_file_path=manual_correction_file_path,
            set_type=0,  # train
            for_adapt=True,
        )
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for epoch in range(n_epochs):

        epoch_loss = 0.
        epoch_accuracy = 0.
        mean_accuracy = 0.
        mean_loss = 0.

        val_accuracy = 0.
        val_loss = 0.

        with Progress() as progress:
            train_task = progress.add_task("[green]Training...", total=len(train_dataset))
            model.train()
            for step, ((data, bc), _, clean_labels) in enumerate(train_dataset):
                data, bc = data.to(device), bc.to(device)
                clean_labels = torch.from_numpy(clean_labels).to(device)

                optimizer.zero_grad()

                logits, _ = model((data, bc))
                loss = F.nll_loss(logits, clean_labels)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                pred_labels = logits.argmax(dim=-1)
                epoch_accuracy += (pred_labels == clean_labels).sum()/pred_labels.size(0)

                mean_accuracy = epoch_accuracy / (step + 1)
                mean_loss = epoch_loss / (step + 1)

                progress.update(train_task, advance=1, description=f"[Epoch {epoch+1}/{n_epochs}] Loss: {mean_loss:.4f}, Accuracy: {mean_accuracy:.4f}")
            progress.stop_task(train_task)
            if remove_progress:
                progress.remove_task(train_task)

            val_task = progress.add_task("[blue]Validating...", total=len(validation_dataset))
            model.eval()
            for step, (((data, bc), _, clean_labels)) in enumerate(validation_dataset):
                data, bc = data.to(device), bc.to(device)
                clean_labels = torch.from_numpy(clean_labels).to(device)

                with torch.no_grad():
                    logits, _ = model((data, bc))
                    loss = F.nll_loss(logits, clean_labels)

                pred_labels = logits.argmax(dim=-1)
                val_loss += loss.item()
                val_accuracy += (pred_labels == clean_labels).sum() / pred_labels.size(0)
                progress.update(val_task, advance=1, description=f"[Epoch {epoch+1}/{n_epochs}] Val Loss: {val_loss/(step+1):.4f}, Val Accuracy: {val_accuracy/(step+1):.4f}")
            progress.stop_task(val_task)
            progress.remove_task(val_task)
            if not remove_progress:
                progress.update(train_task, description=f"[Epoch {epoch+1}/{n_epochs}] Loss: {mean_loss:.4f}, Accuracy: {mean_accuracy:.4f}, Clean Val Loss: {val_loss/(step+1):.4f}, Clean Val Accuracy: {val_accuracy/(step+1):.4f}")
    
    cprint(f"Training completed for {n_epochs} epochs.", "green")
    cprint(f"Computing test set metrics ...")

    with Progress() as progress:
        train_task = progress.add_task("[green]Evaluating on Test Set...", total=len(test_dataset))
        model.eval()
        test_accuracy = 0.
        test_loss = 0.

        for step, (((data, bc), _, clean_labels)) in enumerate(test_dataset):
            data, bc = data.to(device), bc.to(device)
            clean_labels = torch.from_numpy(clean_labels).to(device)

            with torch.no_grad():
                logits, _ = model((data, bc))
                loss = F.nll_loss(logits, clean_labels)

            pred_labels = logits.argmax(dim=-1)
            test_loss += loss.item()
            test_accuracy += (pred_labels == clean_labels).sum() / pred_labels.size(0)

            progress.update(train_task, advance=1, description=f"[Test] Loss: {test_loss/(step+1):.4f}, Accuracy: {test_accuracy/(step+1):.4f}")
                

def train_with_ignore_corrected(
    data_path,
    manual_correction_file_path,
    auto_correction_file_path,
    seed=42,
    n_epochs=20,
    remove_progress=False
):
    lr = 0.001
    weight_decay = 1e-4

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = PartNetGraspWithFilterDataset(
        path_to_zip=data_path,
        correction_file_path=auto_correction_file_path,
        set_type=0,  # train
    )

    validation_dataset = PartNetGraspDataset(
        path_to_zip=data_path,
        correction_file_path=manual_correction_file_path,
        set_type=1,  # validation
    )

    test_dataset = PartNetGraspDataset(
        path_to_zip=data_path,
        correction_file_path=manual_correction_file_path,
        set_type=2,  # test
    )

    model = SegImcnn(
        adapt_data=PartNetGraspDataset(
            path_to_zip=data_path,
            correction_file_path=manual_correction_file_path,
            set_type=0,  # train
            for_adapt=True,
        )
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for epoch in range(n_epochs):

        epoch_loss = 0.
        epoch_accuracy = 0.
        mean_accuracy = 0.
        mean_loss = 0.

        val_accuracy = 0.
        val_loss = 0.

        with Progress() as progress:
            train_task = progress.add_task("[green]Training...", total=len(train_dataset))
            model.train()
            for step, ((data, bc), labels, use_verts_mask) in enumerate(train_dataset):
                data, bc = data.to(device), bc.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                pred = model((data, bc))[0]
                loss = F.nll_loss(pred, labels, reduction='none')
                loss = loss[use_verts_mask].mean()

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                pred_labels = pred.argmax(dim=-1)
                epoch_accuracy += (pred_labels[use_verts_mask] == labels[use_verts_mask]).sum() / use_verts_mask.sum()
                mean_accuracy = epoch_accuracy / (step + 1)
                mean_loss = epoch_loss / (step + 1)

                progress.update(train_task, advance=1, description=f"[Epoch {epoch+1}/{n_epochs}] Loss: {mean_loss:.4f}, Accuracy: {mean_accuracy:.4f}")
            progress.stop_task(train_task)
            if remove_progress:
                progress.remove_task(train_task)

            val_task = progress.add_task("[blue]Validating...", total=len(validation_dataset))
            model.eval()
            for step, (((data, bc), _, clean_labels)) in enumerate(validation_dataset):
                data, bc = data.to(device), bc.to(device)
                clean_labels = torch.from_numpy(clean_labels).to(device)

                with torch.no_grad():
                    logits, _ = model((data, bc))
                    loss = F.nll_loss(logits, clean_labels)

                pred_labels = logits.argmax(dim=-1)
                val_loss += loss.item()
                val_accuracy += (pred_labels == clean_labels).sum() / pred_labels.size(0)
                progress.update(val_task, advance=1, description=f"[Epoch {epoch+1}/{n_epochs}] Val Loss: {val_loss/(step+1):.4f}, Val Accuracy: {val_accuracy/(step+1):.4f}")
            progress.stop_task(val_task)
            progress.remove_task(val_task)
            if not remove_progress:
                progress.update(train_task, description=f"[Epoch {epoch+1}/{n_epochs}] Loss: {mean_loss:.4f}, Accuracy: {mean_accuracy:.4f}, Clean Val Loss: {val_loss/(step+1):.4f}, Clean Val Accuracy: {val_accuracy/(step+1):.4f}")
        
    cprint(f"Training completed for {n_epochs} epochs.", "green")
    cprint(f"Computing test set metrics ...")

    with Progress() as progress:
        train_task = progress.add_task("[green]Evaluating on Test Set...", total=len(test_dataset))
        model.eval()
        test_accuracy = 0.
        test_loss = 0.

        for step, (((data, bc), _, clean_labels)) in enumerate(test_dataset):
            data, bc = data.to(device), bc.to(device)
            clean_labels = torch.from_numpy(clean_labels).to(device)

            with torch.no_grad():
                logits, _ = model((data, bc))
                loss = F.nll_loss(logits, clean_labels)

            pred_labels = logits.argmax(dim=-1)
            test_loss += loss.item()
            test_accuracy += (pred_labels == clean_labels).sum() / pred_labels.size(0)

            progress.update(train_task, advance=1, description=f"[Test] Loss: {test_loss/(step+1):.4f}, Accuracy: {test_accuracy/(step+1):.4f}")


if __name__ == "__main__":

    # every =       [ 5,  5,  5, 10,  5,  5,  5,  5]
    # start_clean = [ 5,  5, 10, 10, 15, 20, 25, 30]
    # k_outlier =   [32, 32, 32, 32, 32, 32, 32, 32]
    # k_cc =        [ 4,  5,  5,  5,  5,  5,  5,  5]
    # zeta =        [.5, .5, .5, .5, .5, .5, .5, .5] 

    # for i in range(len(every)):
    #     params = f"{start_clean[i]}_{every[i]}_{k_outlier[i]}_{k_cc[i]}"
    #     cprint(f"\n\n\n############### Running with params: {params} #########################", "blue")

    #     data_path = "../../SegLabelCorrection/data/partnet_grasp/partnet_grasp.zip"
    #     manual_correction_file_path = "../improve_mesh_segmentation/data_correction/partnet_correction.csv"
    #     auto_correction_file_path = f"./imcnn_corrections_{params}.csv"

    #     cprint("Training with corrected labels...", "yellow")
    #     train_with_corrected(
    #         data_path=data_path,
    #         manual_correction_file_path=manual_correction_file_path,
    #         auto_correction_file_path=auto_correction_file_path,
    #         seed=42,
    #         n_epochs=5,
    #         remove_progress=True
    #     )
    #     cprint("\nTraining with ignored corrected labels...", "yellow")
    #     train_with_ignore_corrected(
    #         data_path=data_path,
    #         manual_correction_file_path=manual_correction_file_path,
    #         auto_correction_file_path=auto_correction_file_path,
    #         seed=42,
    #         n_epochs=5,
    #         remove_progress=True
    #     )

    every =       5
    start_clean = 15
    k_outlier =   32
    k_cc = [
        10, 
        25, 
        50,
        100,
        # 250,
        # 500,
        # 1000,
        # 2000,
        # 3000,
        # 4000,
        # 5000
    ]
    zeta =        .5

    for i in range(len(k_cc)):
        params = f"{start_clean}_{every}_{k_outlier}_{k_cc[i]}"
        cprint(f"\n\n\n############### Running with params: {params} #########################", "blue")

        data_path = "../../SegLabelCorrection/data/partnet_grasp/partnet_grasp.zip"
        manual_correction_file_path = "../improve_mesh_segmentation/data_correction/partnet_correction.csv"
        auto_correction_file_path = f"./imcnn_corrections_{params}.csv"

        cprint("Training with corrected labels...", "yellow")
        train_with_corrected(
            data_path=data_path,
            manual_correction_file_path=manual_correction_file_path,
            auto_correction_file_path=auto_correction_file_path,
            seed=42,
            n_epochs=5,
            remove_progress=True
        )
        cprint("\nTraining with ignored corrected labels...", "yellow")
        train_with_ignore_corrected(
            data_path=data_path,
            manual_correction_file_path=manual_correction_file_path,
            auto_correction_file_path=auto_correction_file_path,
            seed=42,
            n_epochs=5,
            remove_progress=True
        )