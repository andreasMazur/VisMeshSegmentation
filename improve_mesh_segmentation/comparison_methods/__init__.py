from improve_mesh_segmentation.comparison_methods.confident_learning import (
    confident_learning_faust,
    confident_learning_partnet,
)
from improve_mesh_segmentation.comparison_methods.ensemble_majority import (
    ensemble_majority_faust,
    ensemble_majority_partnet,
)
from improve_mesh_segmentation.comparison_methods.global_methods import (
    global_kmeans_observed_faust,
    global_kmeans_observed_partnet,
    global_knn_observed_faust,
    global_knn_observed_partnet,
    global_lvq_observed_faust,
    global_lvq_observed_partnet,
    global_supervised_kmeans_observed_faust,
    global_supervised_kmeans_observed_partnet,
)
from improve_mesh_segmentation.comparison_methods.local_methods import (
    local_kmeans_es_faust,
    local_kmeans_es_partnet,
    local_kmeans_umap_faust,
    local_kmeans_umap_partnet,
    local_knn_es_faust,
    local_knn_es_partnet,
    local_knn_umap_faust,
    local_knn_umap_partnet,
)
from improve_mesh_segmentation.comparison_methods.model_utils import (
    ensure_faust_model,
    ensure_partnet_model,
)
