from improve_mesh_segmentation.comparison_methods.embedding_utils import get_embeddings_labels_preds_with_idx_map
from improve_mesh_segmentation.comparison_methods.helpers import method_logs_dir
from improve_mesh_segmentation.comparison_methods.label_correction import (
    run_global_label_correction_faust,
    run_global_label_correction_partnet,
)
from improve_mesh_segmentation.comparison_methods.model_utils import (
    get_device,
    load_faust_dataset,
    load_faust_model,
    load_partnet_dataset,
    load_partnet_model,
)


def _run_global_partnet(
    method_name, data_path, model_path, experiment_directory, method_parameter, mesh_range, training_epochs=10,
):
    device = get_device()
    imcnn, classification_head = load_partnet_model(
        model_path, data_path, device=device, n_epochs=training_epochs,
    )
    og_dataset = load_partnet_dataset(data_path, set_type=3)
    embs, labels, preds, mesh_map, _, unc_dict = get_embeddings_labels_preds_with_idx_map(
        og_dataset, imcnn, classification_head, mesh_range, device=str(device),
    )
    output_dir = method_logs_dir(experiment_directory, method_name)
    return run_global_label_correction_partnet(
        method_name, embs, labels, preds, mesh_map, unc_dict, output_dir, method_parameter,
    )


def _run_global_faust(
    method_name,
    dataset_path,
    segmentation_labels_path,
    logging_dir,
    noise_level,
    model_path,
    experiment_directory,
    dataset_id,
    method_parameter,
    training_epochs=10,
):
    device = get_device()
    imcnn, classification_head = load_faust_model(
        model_path, dataset_path, segmentation_labels_path, logging_dir, noise_level,
        device=device, n_epochs=training_epochs,
    )
    og_dataset = load_faust_dataset(
        dataset_path, segmentation_labels_path, logging_dir, noise_level, set_type=3, device=device,
    )
    from improve_mesh_segmentation.comparison_methods.embedding_utils import get_embeddings_labels_preds_with_idx_map_faust
    embs, labels, preds, mesh_map, _, unc_dict = get_embeddings_labels_preds_with_idx_map_faust(
        og_dataset, imcnn, classification_head, device=str(device),
    )
    output_dir = method_logs_dir(experiment_directory, method_name, dataset_id=dataset_id)
    return run_global_label_correction_faust(
        method_name, embs, labels, preds, mesh_map, unc_dict, output_dir, method_parameter,
    )


def global_kmeans_observed_partnet(
    data_path, model_path, experiment_directory, method_parameter, mesh_range=range(0, 100), training_epochs=10,
):
    return _run_global_partnet(
        "global_kmeans_observed", data_path, model_path, experiment_directory, method_parameter, mesh_range,
        training_epochs=training_epochs,
    )


def global_knn_observed_partnet(
    data_path, model_path, experiment_directory, method_parameter, mesh_range=range(0, 100), training_epochs=10,
):
    return _run_global_partnet(
        "global_knn_observed", data_path, model_path, experiment_directory, method_parameter, mesh_range,
        training_epochs=training_epochs,
    )


def global_lvq_observed_partnet(
    data_path, model_path, experiment_directory, method_parameter, mesh_range=range(0, 100), training_epochs=10,
):
    return _run_global_partnet(
        "global_lvq_observed", data_path, model_path, experiment_directory, method_parameter, mesh_range,
        training_epochs=training_epochs,
    )


def global_supervised_kmeans_observed_partnet(
    data_path, model_path, experiment_directory, method_parameter, mesh_range=range(0, 100), training_epochs=10,
):
    return _run_global_partnet(
        "global_supervised_kmeans_observed", data_path, model_path, experiment_directory, method_parameter, mesh_range,
        training_epochs=training_epochs,
    )


def global_kmeans_observed_faust(
    dataset_path, segmentation_labels_path, logging_dir, noise_level, model_path,
    experiment_directory, dataset_id, method_parameter, training_epochs=10,
):
    return _run_global_faust(
        "global_kmeans_observed", dataset_path, segmentation_labels_path, logging_dir, noise_level,
        model_path, experiment_directory, dataset_id, method_parameter, training_epochs=training_epochs,
    )


def global_knn_observed_faust(
    dataset_path, segmentation_labels_path, logging_dir, noise_level, model_path,
    experiment_directory, dataset_id, method_parameter, training_epochs=10,
):
    return _run_global_faust(
        "global_knn_observed", dataset_path, segmentation_labels_path, logging_dir, noise_level,
        model_path, experiment_directory, dataset_id, method_parameter, training_epochs=training_epochs,
    )


def global_lvq_observed_faust(
    dataset_path, segmentation_labels_path, logging_dir, noise_level, model_path,
    experiment_directory, dataset_id, method_parameter, training_epochs=10,
):
    return _run_global_faust(
        "global_lvq_observed", dataset_path, segmentation_labels_path, logging_dir, noise_level,
        model_path, experiment_directory, dataset_id, method_parameter, training_epochs=training_epochs,
    )


def global_supervised_kmeans_observed_faust(
    dataset_path, segmentation_labels_path, logging_dir, noise_level, model_path,
    experiment_directory, dataset_id, method_parameter, training_epochs=10,
):
    return _run_global_faust(
        "global_supervised_kmeans_observed", dataset_path, segmentation_labels_path, logging_dir, noise_level,
        model_path, experiment_directory, dataset_id, method_parameter, training_epochs=training_epochs,
    )
