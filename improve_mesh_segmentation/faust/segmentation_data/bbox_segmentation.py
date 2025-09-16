from geoconv_examples.mpi_faust.data.preprocess_faust import get_file_number

import os
import trimesh
import numpy as np
import matplotlib.cm as cm


def define_bbox(start_pos, height, width, depth):
    """Create a bounding box mesh.

    Parameters
    ----------
    start_pos: np.ndarray | list
        The starting position of the bounding box.
    height: float
        The height of the bounding box.
    width: float
        The width of the bounding box.
    depth: float
        The depth of the bounding box.

    Returns
    -------
    trimesh.Trimesh:
        The bounding box mesh.
    """
    if not isinstance(start_pos, np.ndarray):
        start_pos = np.array(start_pos)
    bbox = trimesh.creation.box(extents=(width, height, depth))
    bbox.apply_translation(start_pos)
    return bbox


def bbox_segmentation(registration_path, segmentation_labels_filename, verbose=False):
    """Use bounding boxes to segment the FAUST meshes into parts.

    Parameters
    ----------
    registration_path: str
        Path to the directory containing the registered FAUST meshes.
    segmentation_labels_filename: str
        Path to the output numpy file to save the segmentation labels.
    verbose: bool
        Whether to visualize the segmentation process.
    """
    paths_reg_meshes = os.listdir(registration_path)
    paths_reg_meshes.sort(key=get_file_number)
    paths_reg_meshes = [f for f in paths_reg_meshes if f.endswith(".ply")]

    # Right Arm
    z = 0.55
    x = 0.095
    y = -0.215
    bbox_right_arm = define_bbox([y, z, x], 0.55, 0.3, 0.275)

    # Left Arm
    z = 0.525
    x = 0.095
    y = 0.3
    bbox_left_arm = define_bbox([y, z, x], 0.55, 0.3, 0.37)

    # Torso
    z = 0.008
    x = 0.075
    y = 0.0
    bbox_torso = define_bbox([y, z, x], 0.725, 0.5, 0.275)

    # Head
    z = 0.52
    x = 0.171
    y = 0.05
    bbox_head = define_bbox([y, z, x], 0.299, 0.2, 0.275)

    # Right hand
    z = 0.95
    x = 0.2
    y = -0.3
    bbox_right_hand = define_bbox([y, z, x], 0.25, 0.25, 0.15)

    # Left hand
    z = 0.9
    x = 0.25
    y = 0.4
    bbox_left_hand = define_bbox([y, z, x], 0.2, 0.25, 0.15)

    # Right leg
    z = -0.84
    x = 0.02
    y = -0.1
    bbox_right_leg = define_bbox([y, z, x], 1.0, 0.25, 0.3)

    # Left leg
    z = -0.84
    x = 0.035
    y = 0.16
    bbox_left_leg = define_bbox([y, z, x], 1.0, 0.25, 0.3)

    # Define priority list: Later bboxes overwrite former ones
    head_class = 1
    bbox_list = [
        (bbox_torso, 0),
        (bbox_head, head_class),
        (bbox_right_hand, 2),
        (bbox_left_hand, 3),
        (bbox_right_arm, 4),
        (bbox_left_arm, 5),
        (bbox_right_leg, 6),
        (bbox_left_leg, 7)
    ]

    # Load human mesh
    mesh = trimesh.load_mesh(f"{registration_path}/tr_reg_009.ply")

    if verbose:
        trimesh.Scene(bbox_list + [mesh]).show()

    # Segment the mesh
    segmentation_labels = np.zeros(len(mesh.vertices), dtype=int)
    for bbox, segment in bbox_list[1:]:
        is_within = trimesh.bounds.contains(bbox.bounds, mesh.vertices)
        segmentation_labels[is_within] = segment

    # Improve head segmentation with second mesh
    mesh = trimesh.load_mesh(f"{registration_path}/tr_reg_000.ply")
    z = 0.5
    x = 0.2
    y = 0.09
    bbox_head = define_bbox([y, z, x], 0.299, 0.2, 0.275)
    if verbose:
        trimesh.Scene([bbox_head] + [mesh]).show()
    is_within = trimesh.bounds.contains(bbox_head.bounds, mesh.vertices)
    segmentation_labels[is_within] = head_class

    # Show the segmentation
    if verbose:
        for ply_filename in paths_reg_meshes:  # [9:10]:
            mesh = trimesh.load_mesh(f"{registration_path}/{ply_filename}")
            trimesh.PointCloud(mesh.vertices, colors=cm.get_cmap("tab10")(segmentation_labels)).show()

    np.save(segmentation_labels_filename, segmentation_labels)
