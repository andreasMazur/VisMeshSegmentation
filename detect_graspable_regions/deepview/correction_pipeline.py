from detect_graspable_regions.deepview.deepview import DeepViewSubClass
from detect_graspable_regions.deepview.user_interaction import interactive_seg_correction

import numpy as np


def deep_view_iter(model,
                   signal,
                   bc,
                   coordinates,
                   labels,
                   idx,
                   amount_classes,
                   classes,
                   max_samples,
                   batch_size,
                   embedding_shape,
                   interpolations,
                   lam,
                   resolution,
                   cmap,
                   interactive,
                   title,
                   metric,
                   disc_dist,
                   class_dict,
                   correction_file_name,
                   embed_fn,
                   pred_wrapper):
    """Correct the segmentation labels for one mesh.

    Parameters
    ----------
    model: SegImcnn
        A segmentation IMCNN.
    signal: torch.Tensor
        The signal defined on the shape.
    bc: torch.Tensor
        The barycentric coordinates for the shape.
    coordinates: torch.Tensor
        The 3D-coordinates of the mesh vertices.
    labels: torch.Tensor
        The original ground truth labels for the mesh vertices.
    idx: int
        The mesh index.
    amount_classes: int
        The total amount of available classes
    classes: np.ndarray
        All class labels in an array.
    embedding_shape: tuple
        The shape of the embedding vectors that the model uses to predict vertex probabilities. E.g., (96,) for
        96-dimensional embeddings.
    embed_fn: Callable
        The function that takes the IMCNN and returns the embeddings that the model uses to predict vertex class
        probabilities.
    class_dict: dict
        A dictionary that maps class labels (integers) to their class names (strings).
    batch_size: int
        The batch-size for DeepView.
    max_samples: int
        TODO
    resolution: int
        TODO
    interpolations: int
        TODO
    lam: float
        TODO
    cmap: str
        The colormap to use in DeepView.
    metric: str
        TODO
    disc_dist: bool
        Whether to use the discriminative distance in DeepView.
    interactive: bool
        TODO
    title: str
        The plot title
    correction_file_name: str
        The filename of the CSV-file where to store correction data.
    pred_wrapper: Callable
        A function that returns the predicted probabilities for a mesh vertex of a given IMCNN.
    """
    # Get model embeddings
    embeddings = embed_fn(model, [signal, bc])

    # Determine segments by binning the vertices by their class labels
    coordinates = np.array(coordinates)
    all_segments = [coordinates[np.where(labels == x)[0]] for x in range(amount_classes)]

    # loading the values needed for visualizing the mesh segment
    def data_viz(vertex_idx, pred, gt, cmap):
        interactive_seg_correction(
            shape_idx=idx,
            coordinates=coordinates,
            all_segments=all_segments,
            ground_truth=gt,
            query_indices=vertex_idx,
            cmap=cmap,
            class_dict=class_dict,
            amount_classes=amount_classes,
            file_name=correction_file_name
        )

    imcnn_deepview = DeepViewSubClass(
        lambda x: pred_wrapper(x, model),
        classes,
        max_samples,
        batch_size,
        embedding_shape,
        interpolations,
        lam,
        resolution,
        cmap,
        interactive,
        title,
        metric=metric,
        disc_dist=disc_dist,
        data_viz=data_viz,
        class_dict=class_dict
    )
    imcnn_deepview.add_samples(embeddings, labels)
    imcnn_deepview.show()


def correction_pipeline(model,
                        dataset,
                        embedding_shape,
                        embed_fn,
                        pred_fn,
                        class_dict,
                        signals_are_coordinates=False,
                        batch_size=64,
                        max_samples=7000,
                        resolution=100,
                        interpolations=10,
                        lam=1,
                        cmap=None,
                        metric=None,
                        disc_dist=False,
                        interactive=False,
                        title=None,
                        correction_file_name=None):
    """Use DeepView, IMCNNs and 3D Visualizations to interactively correct segmentation labels for mesh partnet_grasp.

    Parameters
    ----------
    model: SegImcnn
        A segmentation IMCNN.
    dataset: generator
        Generator for the (uncorrected) Partnet-grasp dataset.
    embedding_shape: tuple
        The shape of the embedding vectors that the model uses to predict vertex probabilities. E.g., (96,) for
        96-dimensional embeddings.
    embed_fn: Callable
        The function that takes the IMCNN and returns the embeddings that the model uses to predict vertex class
        probabilities.
    pred_fn: Callable
        The function that takes the IMCNN and returns the probabilities that the model uses to predict vertex classes.
    class_dict: dict
        A dictionary that maps class labels (integers) to their class names (strings).
    signals_are_coordinates: bool
        Whether the signals returned by 'old_dataset' are 3D-coordinates.
    batch_size: int
        The batch-size for DeepView.
    max_samples: int
        TODO
    resolution: int
        TODO
    interpolations: int
        TODO
    lam: float
        TODO
    cmap: str
        The colormap to use in DeepView.
    metric: str
        TODO
    disc_dist: bool
        Whether to use the discriminative distance in DeepView.
    interactive: bool
        TODO
    title: str
        The plot title
    correction_file_name: str
        The filename of the CSV-file where to store correction data.
    """
    # --- Deep View Parameters ----
    amount_classes = len(class_dict)
    classes = np.arange(amount_classes)
    if cmap is None:
        cmap = "tab10"
    if metric is None:
        metric = "euclidean"
    if title is None:
        title = "Default Title"
    if correction_file_name is None:
        correction_file_name = "corrected_labels.csv"
    # -----------------------------

    if signals_are_coordinates:
        for idx, ((signal, bc), labels) in enumerate(dataset):
            deep_view_iter(
                model, signal, bc, signal, labels, idx, amount_classes, classes, max_samples, batch_size,
                embedding_shape, interpolations, lam, resolution, cmap, interactive, title, metric, disc_dist,
                class_dict, correction_file_name, embed_fn, pred_fn
            )
    else:
        for idx, ((signal, bc, coordinates), labels) in enumerate(dataset):
            deep_view_iter(
                model, signal, bc, coordinates, labels, idx, amount_classes, classes, max_samples, batch_size,
                embedding_shape, interpolations, lam, resolution, cmap, interactive, title, metric, disc_dist,
                class_dict, correction_file_name, embed_fn, pred_fn
            )
