from detect_graspable_regions.partnet_grasp.sampling.utils import load_obj
from detect_graspable_regions.partnet_grasp.sampling.constants import CAT2SYNSET

from scipy.spatial.distance import cdist

import trimesh
import numpy as np
import os
import os.path as osp


def manual_filter_mesh(
        base_partnet_path: str,
        aligned_shapenet_path: str,
        anno_id: str,
        cat_name: str,
        model_id: str) -> bool:

    input_objs_dir = osp.join(base_partnet_path, anno_id, 'objs')

    if cat_name not in CAT2SYNSET.keys():
        raise ValueError(f"Category '{cat_name}' not in dictionary map to shapenet!")

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

    shapenet_dir = osp.join(aligned_shapenet_path, CAT2SYNSET[cat_name], model_id)
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

    shapenet_pts = trimesh.sample.sample_surface(shapenet_mesh, 2000)[0]

    dist_mat = cdist(shapenet_pts, partnet_pts)
    chamfer_dist = dist_mat.min(0).mean() + dist_mat.min(1).mean()

    print(f"Chamfer Distance: {chamfer_dist}")

    mesh = trimesh.util.concatenate([partnet_mesh, shapenet_mesh])
    colors = np.concatenate(
        [
            np.repeat([[0, 255, 0, 255]], len(partnet_mesh.vertices)).reshape(4, -1).T,
            np.repeat([[255, 127, 0, 255]], len(shapenet_mesh.vertices)).reshape(4, -1).T
        ]
    )
    print(colors.shape, len(mesh.vertices))
    pc = trimesh.PointCloud(vertices=mesh.vertices, colors=colors)
    pc.show()

    ans = ''
    valid = {'yes': True, 'y': True, 'no': False, 'n': False}
    while ans not in valid.keys():
        ans = input("Use mesh? [y/n]: ")
    return valid[ans]