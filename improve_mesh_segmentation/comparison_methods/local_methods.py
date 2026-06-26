from improve_mesh_segmentation.comparison_methods import filter_methods_faust as faust_filters
from improve_mesh_segmentation.comparison_methods import filter_methods_partnet as partnet_filters
from improve_mesh_segmentation.comparison_methods.helpers import method_logs_dir
from improve_mesh_segmentation.comparison_methods.label_correction import run_local_label_correction
from improve_mesh_segmentation.comparison_methods.model_utils import (
    get_device,
    load_faust_dataset,
    load_faust_model,
    load_partnet_dataset,
    load_partnet_model,
)


def _run_local_partnet(
    method_name, data_path, model_path, experiment_directory, method_parameter, use_umap=False, training_epochs=10,
):
    device = get_device()
    imcnn, classification_head = load_partnet_model(
        model_path, data_path, device=device, n_epochs=training_epochs,
    )
    dataset = load_partnet_dataset(data_path, set_type=3)
    filters = partnet_filters
    output_dir = method_logs_dir(experiment_directory, method_name)
    return run_local_label_correction(
        method_name, filters, imcnn, classification_head, dataset, output_dir, method_parameter,
        device=str(device), use_umap=use_umap,
    )


def _run_local_faust(
    method_name, dataset_path, segmentation_labels_path, logging_dir, noise_level, model_path,
    experiment_directory, dataset_id, method_parameter, use_umap=False, training_epochs=10,
):
    device = get_device()
    imcnn, classification_head = load_faust_model(
        model_path, dataset_path, segmentation_labels_path, logging_dir, noise_level,
        device=device, n_epochs=training_epochs,
    )
    dataset = load_faust_dataset(
        dataset_path, segmentation_labels_path, logging_dir, noise_level, set_type=3, device=device,
    )
    filters = faust_filters
    output_dir = method_logs_dir(experiment_directory, method_name, dataset_id=dataset_id)
    return run_local_label_correction(
        method_name, filters, imcnn, classification_head, dataset, output_dir, method_parameter,
        device=str(device), use_umap=use_umap,
    )


def local_kmeans_umap_partnet(data_path, model_path, experiment_directory, method_parameter, training_epochs=10):
    """PartNet: DeepView k-Means (UMAP-based embedding from DeepView)."""
    return _run_local_partnet(
        "deepview_kmeans_observed", data_path, model_path, experiment_directory, method_parameter,
        training_epochs=training_epochs,
    )


def local_knn_umap_partnet(data_path, model_path, experiment_directory, method_parameter, training_epochs=10):
    """PartNet: DeepView k-NN (UMAP-based embedding from DeepView)."""
    return _run_local_partnet(
        "deepview_knn_observed", data_path, model_path, experiment_directory, method_parameter,
        training_epochs=training_epochs,
    )


def local_kmeans_es_partnet(data_path, model_path, experiment_directory, method_parameter, training_epochs=10):
    return _run_local_partnet(
        "iterative_kmeans_observed", data_path, model_path, experiment_directory, method_parameter,
        training_epochs=training_epochs,
    )


def local_knn_es_partnet(data_path, model_path, experiment_directory, method_parameter, training_epochs=10):
    return _run_local_partnet(
        "iterative_knn_observed", data_path, model_path, experiment_directory, method_parameter,
        training_epochs=training_epochs,
    )


def local_kmeans_umap_faust(
    dataset_path, segmentation_labels_path, logging_dir, noise_level, model_path,
    experiment_directory, dataset_id, method_parameter, training_epochs=10,
):
    """FAUST: direct UMAP k-Means."""
    return _run_local_faust(
        "umap_kmeans_observed", dataset_path, segmentation_labels_path, logging_dir, noise_level,
        model_path, experiment_directory, dataset_id, method_parameter, use_umap=True,
        training_epochs=training_epochs,
    )


def local_knn_umap_faust(
    dataset_path, segmentation_labels_path, logging_dir, noise_level, model_path,
    experiment_directory, dataset_id, method_parameter, training_epochs=10,
):
    """FAUST: direct UMAP k-NN."""
    return _run_local_faust(
        "umap_knn_observed", dataset_path, segmentation_labels_path, logging_dir, noise_level,
        model_path, experiment_directory, dataset_id, method_parameter, use_umap=True,
        training_epochs=training_epochs,
    )


def local_kmeans_es_faust(
    dataset_path, segmentation_labels_path, logging_dir, noise_level, model_path,
    experiment_directory, dataset_id, method_parameter, training_epochs=10,
):
    return _run_local_faust(
        "iterative_kmeans_observed", dataset_path, segmentation_labels_path, logging_dir, noise_level,
        model_path, experiment_directory, dataset_id, method_parameter, training_epochs=training_epochs,
    )


def local_knn_es_faust(
    dataset_path, segmentation_labels_path, logging_dir, noise_level, model_path,
    experiment_directory, dataset_id, method_parameter, training_epochs=10,
):
    return _run_local_faust(
        "iterative_knn_observed", dataset_path, segmentation_labels_path, logging_dir, noise_level,
        model_path, experiment_directory, dataset_id, method_parameter, training_epochs=training_epochs,
    )
