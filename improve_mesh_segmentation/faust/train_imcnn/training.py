from improve_mesh_segmentation.faust.segmentation_data.faust_segmentation_dataset import FaustSegmentationDataset
from improve_mesh_segmentation.training.imcnn import SegImcnn
from improve_mesh_segmentation.training.train_logging import log_training

from torch import nn

import torch
import os


def training(dataset_path,
             segmentation_labels_path,
             logging_dir,
             device,
             n_epochs,
             skip_validation=False,
             skip_testing=False,
             verbose=False):
    os.makedirs(logging_dir, exist_ok=True)

    model = SegImcnn(
        adapt_data=FaustSegmentationDataset(
            path_to_zip=dataset_path,
            path_to_segmentation_labels=segmentation_labels_path,
            set_type=0,
            only_signal=True,
            device=device
        ),
        signal_dim=544,
        kernel_size=(3, 6),
        segmentation_classes=8,
        template_radius=0.027744965069279016,
        layer_conf=[(32, 6), (32, 6)]
    ).to(device)
    train_data = FaustSegmentationDataset(
        path_to_zip=dataset_path,
        path_to_segmentation_labels=segmentation_labels_path,
        set_type=0,
        only_signal=False,
        device=device
    )
    val_data = FaustSegmentationDataset(
        path_to_zip=dataset_path,
        path_to_segmentation_labels=segmentation_labels_path,
        set_type=1,
        only_signal=False,
        device=device
    )
    test_data = FaustSegmentationDataset(
        path_to_zip=dataset_path,
        path_to_segmentation_labels=segmentation_labels_path,
        set_type=1,
        only_signal=False,
        device=device
    )

    train_hist = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "test_loss": [],
        "test_accuracy": []
    }
    for epoch in range(n_epochs):
        # Reset generators for new epoch
        train_data.reset()
        val_data.reset()

        # Training
        epoch_train_hist = model.train_loop(
            dataset=train_data,
            loss_fn=nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam(model.parameters()),
            verbose=True,
            epoch=epoch
        )
        train_hist["train_loss"].append(float(epoch_train_hist["epoch_loss"].detach()))
        train_hist["train_accuracy"].append(float(epoch_train_hist["epoch_accuracy"].detach()))

        # Validation
        if not skip_validation:
            epoch_val_hist = model.validation_loop(
                dataset=val_data,
                loss_fn=nn.CrossEntropyLoss(),
                verbose=True
            )
            train_hist["val_loss"].append(float(epoch_val_hist["val_epoch_loss"].detach()))
            train_hist["val_accuracy"].append(float(epoch_val_hist["val_epoch_accuracy"].detach()))

    # Testing
    if not skip_testing:
        epoch_test_hist = model.validation_loop(
            dataset=test_data,
            loss_fn=nn.CrossEntropyLoss(),
            verbose=False
        )
        train_hist["test_loss"].append(float(epoch_test_hist["val_epoch_loss"].detach()))
        train_hist["test_accuracy"].append(float(epoch_test_hist["val_epoch_accuracy"].detach()))

    if logging_dir is not None:
        log_training(model, train_hist, logging_dir, skip_validation, skip_testing, verbose=verbose)

    return model, train_hist
