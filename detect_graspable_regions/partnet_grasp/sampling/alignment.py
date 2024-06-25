from detect_graspable_regions.partnet_grasp.sampling.utils import load_obj, rot_mat
from detect_graspable_regions.partnet_grasp.sampling.constants import CAT2SYNSET

from scipy.spatial.distance import cdist

import os
import os.path as osp
import numpy as np
import trimesh


def align_to_partnet(
        base_partnet_path: str,
        base_shapenet_path: str,
        anno_id: str,
        cat_name: str,
        model_id: str,
        aligned_shapenet_target_path: str = None,
        verbose: bool = False,
        show_meshes: bool = False
    ) -> None:
    """Saves a ShapeNet mesh that was heuristically aligned to the corresponding PartNet mesh for further processing.

    Args:
        base_partnet_path (str): Path to root PartNet folder
        base_shapenet_path (str): Path to root ShapeNet folder
        anno_id (str): id of the model in PartNet
        cat_name (str): category the model belongs to
        model_id (str): hash (id) of the ShapeNet mesh
        aligned_shapenet_target_path (str, optional): Root path to where the mesh should be saved to.
         Defaults to ./PartNetAligned.
        verbose (bool, optional): Whether to print model statistics after transformations were applied.
         Defaults to False.
        show_meshes (bool, optional): Whether overlayed meshes should be displayed for inspection purposes.
         Defaults to False.

    Raises:
        ValueError: If the category is unknown
        ValueError: If the ShapeNet directory does not exist
        ValueError: If the ShapeNet model could not be loaded
    """

    if aligned_shapenet_target_path is None:
        aligned_shapenet_target_path = './PartNetAligned'

    os.makedirs(aligned_shapenet_target_path, exist_ok=True)
    input_objs_dir = osp.join(base_partnet_path, anno_id, 'objs')

    if cat_name not in CAT2SYNSET.keys():
        raise ValueError(f"Category '{cat_name}' not in dictionary map to shapenet!")

    def shapenet_trans(verts: np.ndarray) -> np.ndarray:
        """Computes the transform from the given mesh to a unit xy-diagonal and axis centered mesh
           and returns the inverted transform.

        Args:
            verts (np.ndarray): Vertices of the mesh to get the transform to.

        Returns:
            np.ndarray: Homogeneus transformation matrix from a unit xy-diagonal and centered mesh to the given mesh.
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
        z_len = z_max - z_min

        scale = np.sqrt(x_len**2 + y_len**2)
        # scale = max(x_len, max(y_len, z_len))
        # scale = 1/max_dim

        trans = np.array([
            [0, 0, 1.0 / scale, -x_center/scale],
            [0, 1.0 / scale, 0, -y_center/scale],
            [-1/scale, 0, 0, -z_center/scale],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        trans = np.linalg.inv(trans)
        return trans

    def transform_verts(trans: np.ndarray,
                        in_verts: np.ndarray,
                        in_faces: np.ndarray,
                        rot: np.ndarray = None) -> trimesh.Trimesh:
        """Applies homogeneous transformations to a given set of vertices and returns the resulting mesh.

        Args:
            trans (np.ndarray): Homogeneus transformation matrix.
            in_verts (np.ndarray): Vertices of the mesh to transform.
            in_faces (np.ndarray): Faces of the mesh to transform.
            rot (np.ndarray, optional): Additional rotation to apply to the mesh. Defaults to None.

        Returns:
            trimesh.Trimesh: Transformed mesh.
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

    vs = []
    fs = []
    vid = 0
    for item in os.listdir(input_objs_dir):
        if item.endswith('.obj'):
            cur_vs, cur_fs = load_obj(osp.join(input_objs_dir, item))
            vs.append(cur_vs)
            fs.append(cur_fs + vid)
            vid += cur_vs.shape[0]

    v_arr = np.concatenate(vs, axis=0)
    v_arr_ori = np.array(v_arr, dtype=np.float32)
    f_arr = np.concatenate(fs, axis=0)
    tmp = np.array(v_arr[:, 0], dtype=np.float32)
    v_arr[:, 0] = v_arr[:, 2]
    v_arr[:, 2] = -tmp
    trans = shapenet_trans(v_arr)

    shapenet_dir = osp.join(base_shapenet_path, CAT2SYNSET[cat_name], model_id)
    out_file = osp.join(aligned_shapenet_target_path, CAT2SYNSET[cat_name], model_id, 'training', 'model_normalized.obj')
    os.makedirs(osp.join(aligned_shapenet_target_path, CAT2SYNSET[cat_name], model_id, 'training'), exist_ok=True)
    if not osp.exists(shapenet_dir):
        raise ValueError(f"Shapenet dir {shapenet_dir} does not exist!")
    tmp_mesh = trimesh.load(osp.join(shapenet_dir, 'training', 'model_normalized.obj'))

    if isinstance(tmp_mesh, trimesh.Scene):
        shapenet_mesh = trimesh.util.concatenate(
            tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces) for g in tmp_mesh.geometry.values())
        )
    elif isinstance(tmp_mesh, trimesh.Trimesh):
        shapenet_mesh = trimesh.Trimesh(vertices=tmp_mesh.vertices, faces=tmp_mesh.faces)
    else:
        raise ValueError("ERROR: failed to correctly load shapenet mesh!")

    # Test dist
    partnet_mesh = trimesh.Trimesh(vertices=v_arr_ori, faces=f_arr-1)
    partnet_pts = trimesh.sample.sample_surface(partnet_mesh, 2000)[0]

    shapenet_mesh_t = transform_verts(trans, shapenet_mesh.vertices, shapenet_mesh.faces)
    shapenet_pts = trimesh.sample.sample_surface(shapenet_mesh_t, 2000)[0]

    dist_mat = cdist(shapenet_pts, partnet_pts)
    chamfer_dist = dist_mat.min(0).mean() + dist_mat.min(1).mean()

    if chamfer_dist > 0.1:
        print(f"Misalignment detected ({chamfer_dist}), trying rotation")
        shapenet_mesh_t = transform_verts(trans, shapenet_mesh.vertices, shapenet_mesh.faces, rot_mat(np.pi/2))
        shapenet_pts  = trimesh.sample.sample_surface(shapenet_mesh_t, 2000)[0]
        dist_mat = cdist(shapenet_pts, partnet_pts)
        chamfer_dist = dist_mat.min(0).mean() + dist_mat.min(1).mean()
    print(f"Chamfer Distance: {chamfer_dist}")

    # Align centers
    verts = partnet_mesh.vertices
    p_x_min = np.min(verts[:, 0])
    p_x_max = np.max(verts[:, 0])
    p_x_len = p_x_max - p_x_min
    p_y_min = np.min(verts[:, 1])
    p_y_max = np.max(verts[:, 1])
    p_y_len = p_y_max - p_y_min
    p_z_min = np.min(verts[:, 2])
    p_z_max = np.max(verts[:, 2])
    p_z_len = p_z_max - p_z_min

    verts = shapenet_mesh_t.vertices
    s_x_min = np.min(verts[:, 0])
    s_x_max = np.max(verts[:, 0])
    s_x_len = s_x_max - s_x_min
    s_y_min = np.min(verts[:, 1])
    s_y_max = np.max(verts[:, 1])
    s_y_len = s_y_max - s_y_min
    s_z_min = np.min(verts[:, 2])
    s_z_max = np.max(verts[:, 2])
    s_z_len = s_z_max - s_z_min

    f = p_x_len / s_x_len
    scale = np.array([
        [f, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, f, 0],
        [0, 0, 0, 0]
    ])

    shapenet_mesh_t = transform_verts(scale, shapenet_mesh_t.vertices, shapenet_mesh_t.faces)
    shapenet_pts  = trimesh.sample.sample_surface(shapenet_mesh_t, 2000)[0]
    dist_mat = cdist(shapenet_pts, partnet_pts)
    chamfer_dist = dist_mat.min(0).mean() + dist_mat.min(1).mean()
    print(f"Chamfer Distance new scale: {chamfer_dist}")

    verts = shapenet_mesh_t.vertices
    s_x_min = np.min(verts[:, 0])
    s_x_max = np.max(verts[:, 0])
    s_x_len = s_x_max - s_x_min
    s_y_min = np.min(verts[:, 1])
    s_y_max = np.max(verts[:, 1])
    s_y_len = s_y_max - s_y_min
    s_z_min = np.min(verts[:, 2])
    s_z_max = np.max(verts[:, 2])
    s_z_len = s_z_max - s_z_min

    offset = np.array([
        [1, 0, 0, p_x_min - s_x_min],
        [0, 1, 0, p_y_min - s_y_min],
        [0, 0, 1, p_z_min - s_z_min],
        [0, 0, 0, 0]
    ])
    shapenet_mesh_t = transform_verts(offset, shapenet_mesh_t.vertices, shapenet_mesh_t.faces)
    shapenet_pts = trimesh.sample.sample_surface(shapenet_mesh_t, 2000)[0]
    dist_mat = cdist(shapenet_pts, partnet_pts)
    chamfer_dist = dist_mat.min(0).mean() + dist_mat.min(1).mean()
    print(f"Chamfer Distance new t: {chamfer_dist}")

    verts = shapenet_mesh_t.vertices
    s_x_min = np.min(verts[:, 0])
    s_x_max = np.max(verts[:, 0])
    s_x_len = s_x_max - s_x_min
    s_y_min = np.min(verts[:, 1])
    s_y_max = np.max(verts[:, 1])
    s_y_len = s_y_max - s_y_min
    s_z_min = np.min(verts[:, 2])
    s_z_max = np.max(verts[:, 2])
    s_z_len = s_z_max - s_z_min

    if verbose:
        print(f"partnet measures:\n{p_x_len}, {p_y_len}, {p_z_len}; {p_z_len**2+p_y_len**2}")
        print(f"shapenet measures:\n{s_x_len}, {s_y_len}, {s_z_len}; {s_y_len**2+s_z_len**2}")

    trimesh.exchange.export.export_mesh(shapenet_mesh_t, out_file, file_type='obj')

    if show_meshes:
        mesh = trimesh.util.concatenate([partnet_mesh, shapenet_mesh_t])
        colors = np.concatenate(
            [
                np.repeat([[0, 255, 0, 255]], len(partnet_mesh.vertices)).reshape(4, -1).T,
                np.repeat([[255, 127, 0, 255]], len(shapenet_mesh_t.vertices)).reshape(4, -1).T
            ]
        )
        print(colors.shape, len(mesh.vertices))
        pc = trimesh.PointCloud(vertices=mesh.vertices, colors=colors)
        pc.show()
