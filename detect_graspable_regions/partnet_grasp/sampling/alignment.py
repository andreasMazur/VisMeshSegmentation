from detect_graspable_regions.partnet_grasp.sampling.utils import rot_mat, ModelHandler, get_chamfer_dist
from detect_graspable_regions.partnet_grasp.sampling.constants import CAT2SYNSET

from io import BytesIO

import os.path as osp
import numpy as np
import trimesh
import zipfile


def get_shapenet_trans(model : ModelHandler) -> np.ndarray:
    """Computes the inverse transformation from a combined PartNet model to a ShapeNet model.

    Parameters
    ----------
    model : ModelHandler
        Handler to the model instance to get the transform to.
    Returns:
    --------
    np.ndarray
        Homogeneus transformation matrix from a unit xy-diagonal and centered mesh to the given mesh.
    """
    def shapenet_trans(verts: np.ndarray) -> np.ndarray:
        """Computes the transform from the given mesh to a unit xy-diagonal and axis centered mesh
        and returns the inverted transform.

        Parameters
        ----------
        verts : np.ndarray
            Vertices of the mesh to get the transform to.

        Returns:
        --------
        np.ndarray
            Homogeneus transformation matrix from a unit xy-diagonal and centered mesh to the given mesh.
        """
        x_min = np.min(verts[:, 0])
        x_max = np.max(verts[:, 0])
        x_center = (x_min + x_max) / 2
        x_len = x_max - x_min
        y_min = np.min(verts[:, 1])
        y_max = np.max(verts[:, 1])
        y_center = (y_min + y_max) / 2
        y_len = y_max - y_min
        z_min = np.min(verts[:, 2])
        z_max = np.max(verts[:, 2])
        z_center = (z_min + z_max) / 2

        scale = np.sqrt(x_len ** 2 + y_len ** 2)

        trans = np.array([
            [0, 0, 1.0 / scale, -x_center / scale],
            [0, 1.0 / scale, 0, -y_center / scale],
            [-1 / scale, 0, 0, -z_center / scale],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        trans = np.linalg.inv(trans)

        return trans

    # combine segments to single object
    vs = []
    for segment, _ in model.partnet_bytes_segments:
        # Read with trimesh
        # (load dict is enough because it already contains parsed vertices)
        vs.append(trimesh.exchange.obj.load_obj(
            segment,
            maintain_order=True,
            skip_materials=True,
            process=False
        )['vertices'])

    # get the shapenet transformation matrix as recommended by
    # the original authors
    v_arr = np.concatenate(vs, axis=0)
    tmp = np.array(v_arr[:, 0], dtype=np.float32)
    v_arr[:, 0] = v_arr[:, 2]
    v_arr[:, 2] = -tmp
    return shapenet_trans(v_arr)


def transform_vertices(trans: np.ndarray,
                       in_verts: np.ndarray,
                       in_faces: np.ndarray,
                       rot: np.ndarray = None) -> trimesh.Trimesh:
    """Applies homogeneous transformations to a given set of vertices and returns the resulting mesh.

    Parameters
    ----------
    trans : np.ndarray
        Homogeneus transformation matrix.
    in_verts : np.ndarray
        Vertices of the mesh to transform.
    in_faces : np.ndarray
        Faces of the mesh to transform.
    rot : np.ndarray, optional
        Additional rotation to apply to the mesh. Defaults to None.

    Returns:
    --------
    trimesh.Trimesh
        Transformed mesh.
    """
    verts = np.array(in_verts, dtype=np.float32)
    verts = np.concatenate([verts, np.ones((verts.shape[0], 1), dtype=np.float32)], axis=1)
    if rot is not None:
        verts = verts @ rot @ (trans.T)
    else:
        verts = verts @ (trans.T)
    verts = verts[:, :3]

    out_mesh = trimesh.Trimesh(vertices=verts, faces=in_faces)
    return out_mesh


def align_to_partnet(model: ModelHandler,
                     aligned_archive: zipfile.ZipFile = None,
                     show_meshes: bool = False) -> None:
    """Saves a ShapeNet mesh that was heuristically aligned to the corresponding PartNet mesh for further processing.

    Parameters
    ----------
    model : ModelHandler
        handler combining ShapeNet and PartNet meta-information.
    aligned_archive : zipfile.ZipFile, optional
        archive the aligned model will be stored in, by default None.
        If None, the archive location will automatically be determined to be in
        the same place as the original archive.
    show_meshes : bool, optional
        Whether to display the meshes for visual inspection of the alignment quality, by default False

    Raises
    ------
    ValueError
        ValueError: If the category is unknown
    """
    

    cat_name = model.meta['model_cat'].lower()
    model_id = model.meta['model_id']

    # Choose destination of aligned dataset
    if aligned_archive is None:
        aligned_archive = zipfile.ZipFile(model.shapenet_dataset.base_path/"ShapeNetAligned.zip", 'a', zipfile.ZIP_DEFLATED)

    # check if category synset is known
    if cat_name not in CAT2SYNSET.keys():
        raise ValueError(f"Category '{cat_name}' not in dictionary map to shapenet!")

    # get transfrom ShapeNet model -> PartNet combined model
    trans = get_shapenet_trans(model)

    # combine all PartNet segments into one scene
    # note that the combined segments are not connected by faces!
    segments = []
    for seg, _ in model.partnet_segments:
        segments.append(seg)
    partnet_full_mesh = trimesh.util.concatenate(
        segments
    )

    out_file = osp.join(CAT2SYNSET[cat_name], model_id, 'training', 'model_normalized.obj')
    out_buf = BytesIO()

    # transform ShapeNet mesh and measure chamfer distance (i.e., pointcloud fit)
    shapenet_mesh_t = transform_vertices(trans, model.shapenet_mesh.vertices, model.shapenet_mesh.faces)
    chamfer_dist = get_chamfer_dist(shapenet_mesh_t, partnet_full_mesh)

    best_verts = shapenet_mesh_t.vertices
    lowest_dist = chamfer_dist

    # distance too big. Try rotating by 90 deg befor transforming, as suggested by PartNet authors
    if chamfer_dist > 0.1:
        print(f"Misalignment detected ({chamfer_dist}), trying rotation")
        shapenet_mesh_t_new = transform_vertices(trans, model.shapenet_mesh.vertices, model.shapenet_mesh.faces, rot_mat(np.pi/2))
        chamfer_dist = get_chamfer_dist(shapenet_mesh_t_new, partnet_full_mesh)

        # if rotation does not help, keep unrotated
        if chamfer_dist < lowest_dist:
            best_verts = shapenet_mesh_t.vertices
            lowest_dist = chamfer_dist
            shapenet_mesh_t = shapenet_mesh_t_new
    print(f"Chamfer Distance: {chamfer_dist}")

    # Scale

    ## Compute PartNet model statistics
    verts = partnet_full_mesh.vertices
    p_x_min = np.min(verts[:, 0])
    p_x_max = np.max(verts[:, 0])
    p_x_len = p_x_max - p_x_min
    p_means = verts.mean(axis=0)

    ## Compute initial ShapeNet model statistics
    verts = shapenet_mesh_t.vertices
    s_x_min = np.min(verts[:, 0])
    s_x_max = np.max(verts[:, 0])
    s_x_len = s_x_max - s_x_min

    f = p_x_len / s_x_len
    scale = np.array([
        [f, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, f, 0],
        [0, 0, 0, 0]
    ])

    shapenet_mesh_t = transform_vertices(scale, shapenet_mesh_t.vertices, shapenet_mesh_t.faces)
    chamfer_dist = get_chamfer_dist(shapenet_mesh_t, partnet_full_mesh)
    if chamfer_dist < lowest_dist:
        best_verts = shapenet_mesh_t.vertices
        lowest_dist = chamfer_dist
    print(f"Chamfer Distance new scale: {chamfer_dist}")

    ## Compute changed ShapeNet model statistics
    verts = shapenet_mesh_t.vertices
    s_means = verts.mean(axis=0)

    # Align centroids
    offset = np.array([
        [1, 0, 0, p_means[0] - s_means[0]],
        [0, 1, 0, p_means[1] - s_means[1]],
        [0, 0, 1, p_means[2] - s_means[2]],
        [0, 0, 0, 0]
    ])
    shapenet_mesh_t = transform_vertices(offset, shapenet_mesh_t.vertices, shapenet_mesh_t.faces)
    chamfer_dist = get_chamfer_dist(shapenet_mesh_t, partnet_full_mesh)
    if chamfer_dist < lowest_dist:
        best_verts = shapenet_mesh_t.vertices
        lowest_dist = chamfer_dist
    print(f"Chamfer Distance new t: {chamfer_dist}")

    best_mesh = trimesh.Trimesh(best_verts, shapenet_mesh_t.faces)
    print(f"Best Chamfer dist: {lowest_dist}")

    # Export best model, i.e., the model with the lowest chamfer distance
    trimesh.exchange.export.export_mesh(best_mesh, out_buf, file_type='obj')
    aligned_archive.writestr(out_file, out_buf.getvalue())

    if show_meshes:
        mesh = trimesh.util.concatenate([partnet_full_mesh, shapenet_mesh_t])
        colors = np.concatenate(
            [
                np.repeat([[0, 255, 0, 255]], len(partnet_full_mesh.vertices)).reshape(4, -1).T,
                np.repeat([[255, 127, 0, 255]], len(shapenet_mesh_t.vertices)).reshape(4, -1).T
            ]
        )
        print(colors.shape, len(mesh.vertices))
        pc = trimesh.PointCloud(vertices=mesh.vertices, colors=colors)
        pc.show()
