import numpy as np
import os.path as osp
import trimesh
import open3d as o3d


def get_recurrent_objs(base_partnet_path, label, anno_id, to_parse):
    out_dict = {}
    if 'children' in to_parse.keys():
        for child in to_parse['children']:
            res = get_recurrent_objs(base_partnet_path, label, anno_id, child)
            out_dict = {**out_dict, **res}
    else:
        for model in to_parse['objs']:
            cur_vs, cur_fs = load_obj(osp.join(base_partnet_path, anno_id, 'objs', f'{model}.obj'))
            o3d_mesh = trimesh.Trimesh(vertices=cur_vs, faces=cur_fs-1).as_open3d
            o3d_mesh = o3d.t.geometry.TriangleMesh.from_legacy(o3d_mesh)
            out_dict[o3d_mesh] = label
    return out_dict


def rot_mat(angle):
    return np.array([
        np.cos(angle), 0, np.sin(angle), 0,
        0, 1, 0, 0,
        -np.sin(angle), 0, np.cos(angle), 0,
        0, 0, 0, 1
    ]).reshape(4, 4)


def load_obj(fn):
    fin = open(fn, 'r')
    lines = [line.rstrip() for line in fin]
    fin.close()

    vertices = []
    faces = []
    for line in lines:
        if line.startswith('v '):
            vertices.append(np.float32(line.split()[1:4]))
        elif line.startswith('f '):
            faces.append(np.int32([item.split('/')[0] for item in line.split()[1:4]]))
    return np.vstack(vertices), np.vstack(faces)
