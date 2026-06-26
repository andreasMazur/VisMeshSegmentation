import os

import torch

from improve_mesh_segmentation.faust.segmentation_data.faust_segmentation_dataset import FaustSegmentationDataset
from improve_mesh_segmentation.faust.train_imcnn.training import training as train_faust_imcnn
from improve_mesh_segmentation.partnet_grasp.dataset import PartNetGraspDataset, processed_partnet_grasp_generator
from improve_mesh_segmentation.training.imcnn import SegImcnn
from improve_mesh_segmentation.training.train_imcnn import train_single_imcnn


FAUST_IMCNN_KWARGS = {
    "signal_dim": 544,
    "kernel_size": (3, 6),
    "segmentation_classes": 8,
    "template_radius": 0.027744965069279016,
    "layer_conf": [(32, 6), (32, 6)],
}


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _require_file(path, description):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{description} not found at {path}")


def ensure_partnet_model(model_path, data_path, n_epochs=10, verbose=False):
    """Train the PartNet-Grasp IMCNN at ``model_path`` if it does not exist yet."""
    if os.path.isfile(model_path):
        return model_path

    _require_file(data_path, "PartNet-Grasp dataset")
    logging_dir = os.path.dirname(model_path)
    os.makedirs(logging_dir, exist_ok=True)

    device = get_device()
    print(f"Model not found at {model_path}. Training PartNet-Grasp IMCNN for {n_epochs} epochs...")
    train_single_imcnn(
        data_path=data_path,
        n_epochs=n_epochs,
        device=str(device),
        logging_dir=logging_dir,
        skip_validation=False,
        skip_testing=False,
        verbose=verbose,
    )

    if not os.path.isfile(model_path):
        raise RuntimeError(f"Training finished but model is still missing at {model_path}")
    return model_path


def ensure_faust_model(
    model_path,
    dataset_path,
    segmentation_labels_path,
    logging_dir,
    noise_level,
    n_epochs=10,
    verbose=False,
):
    """Train the FAUST IMCNN at ``model_path`` if it does not exist yet."""
    if os.path.isfile(model_path):
        return model_path

    _require_file(dataset_path, "FAUST dataset")
    _require_file(segmentation_labels_path, "FAUST segmentation labels")
    os.makedirs(logging_dir, exist_ok=True)

    device = get_device()
    print(
        f"Model not found at {model_path}. "
        f"Training FAUST IMCNN (noise_level={noise_level}) for {n_epochs} epochs..."
    )
    train_faust_imcnn(
        dataset_path=dataset_path,
        segmentation_labels_path=segmentation_labels_path,
        logging_dir=logging_dir,
        device=device,
        n_epochs=n_epochs,
        skip_validation=False,
        skip_testing=False,
        verbose=verbose,
        noise_level=noise_level,
    )

    if not os.path.isfile(model_path):
        raise RuntimeError(f"Training finished but model is still missing at {model_path}")
    return model_path


def load_partnet_model(model_path, data_path, device=None, n_epochs=10, train_if_missing=True, verbose=False):
    if train_if_missing:
        ensure_partnet_model(model_path, data_path, n_epochs=n_epochs, verbose=verbose)

    device = device or get_device()
    imcnn = SegImcnn(adapt_data=PartNetGraspDataset(data_path, set_type=3, only_signal=True))
    imcnn.load_state_dict(torch.load(model_path, map_location=device))
    imcnn.to(device)
    imcnn.eval()
    return imcnn, imcnn.model.output_dense


def load_partnet_dataset(data_path, set_type=3):
    return list(processed_partnet_grasp_generator(data_path, set_type=set_type))


def load_faust_model(
    model_path,
    dataset_path,
    segmentation_labels_path,
    logging_dir,
    noise_level,
    device=None,
    n_epochs=10,
    train_if_missing=True,
    verbose=False,
):
    if train_if_missing:
        ensure_faust_model(
            model_path,
            dataset_path,
            segmentation_labels_path,
            logging_dir,
            noise_level,
            n_epochs=n_epochs,
            verbose=verbose,
        )

    device = device or get_device()
    adapt_data = FaustSegmentationDataset(
        path_to_zip=dataset_path,
        path_to_segmentation_labels=segmentation_labels_path,
        logging_dir=logging_dir,
        set_type=0,
        only_signal=True,
        device=device,
        noise_level=noise_level,
    )
    imcnn = SegImcnn(adapt_data=adapt_data, **FAUST_IMCNN_KWARGS).to(device)
    imcnn.load_state_dict(torch.load(model_path, map_location=device))
    imcnn.eval()
    return imcnn, imcnn.model.output_dense


def load_faust_dataset(dataset_path, segmentation_labels_path, logging_dir, noise_level, set_type=3, device=None):
    device = device or get_device()
    faust_data = FaustSegmentationDataset(
        path_to_zip=dataset_path,
        path_to_segmentation_labels=segmentation_labels_path,
        logging_dir=logging_dir,
        set_type=set_type,
        only_signal=False,
        device=device,
        noise_level=noise_level,
    )
    return list(faust_data.dataset)
