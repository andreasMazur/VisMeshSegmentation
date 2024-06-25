from detect_graspable_regions.partnet_grasp.sampling.constants import CAT2SYNSET, LABLEMAP, COLORMAP, TMPMAP
from detect_graspable_regions.partnet_grasp.sampling.partnet_annotated_mesh import PartNetAnnotatedMesh
from detect_graspable_regions.partnet_grasp.sampling.utils import get_recurrent_objs

import os
import os.path as osp
import trimesh
import open3d as o3d
import numpy as np


def orig_segmentation(
            base_partnet_path: str,
            aligned_shapenet_path: str,
            annot: PartNetAnnotatedMesh,
            out_npz_dir: str,
            verbose: bool = False,
            manual: bool = False
        ) -> None:
    """Computes aligned ShapeNet labels from signed distance to PartNet segments.

    Args:
        base_partnet_path (str): Path to `PartNet/data_vX` dir.
        aligned_shapenet_path (str): Path to aligned ShapeNet partnet_grasp root dir.
        annot (PartNetAnnotatedMesh): Mesh to process.
        out_npz_dir (str): Directory where to save a numpy file containing vertices, faces, and segment labels.
        verbose (bool, optional): Print more info. Defaults to False.
        TODO -- manual docstring

    Raises:
        ValueError: If the category is unknown
        ValueError: If the aligned ShapeNet directory does not exist
        ValueError: If the aligned ShapeNet model could not be loaded
    """

    os.makedirs(out_npz_dir, exist_ok=True)

    anno_id = annot.anno_id
    cat_name = annot.meta['model_cat'].lower()
    model_id = annot.meta['model_id']

    if cat_name not in CAT2SYNSET.keys():
        raise ValueError(f"Category '{cat_name}' not in dictionary map to shapenet!")

    # Load segments as individual meshes
    segments = {}
    for segment in annot.segments:
        try:
            label = LABLEMAP[segment['name']]
        except KeyError:
            print(f"ERROR: Did contain segment class {segment['name']}. Assigning 'other'")
            label = 4

        if verbose:
            print(f"label is: {label} (segment: {segment})")
        tmp = get_recurrent_objs(
            base_partnet_path=base_partnet_path,
            label=label,
            anno_id=anno_id,
            to_parse=segment
        )
        segments = {**segments, **tmp}

    if verbose:
        for m in segments.keys():
            print(f"Mesh type: {type(m)}")

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

    # sadly no cuda support for embree yet , device=o3d.core.Device("CUDA:0"))
    verts = o3d.core.Tensor(shapenet_mesh.vertices, dtype=o3d.core.Dtype.Float32)
    distances = []
    lbls = []
    for mesh, lbl in segments.items():
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh)
        dist = scene.compute_signed_distance(verts)
        distances.append(dist)
        lbls.append(lbl)

    # Get index i.e label with min dist
    res = np.stack(distances, axis=-1).argmin(axis=1)

    labels = []
    colors = []
    for _, vert in enumerate(res):
        colors.append(COLORMAP[TMPMAP[lbls[vert]]])
        labels.append(lbls[vert]-1)

    m_valid = True
    print(f"INFO: vertcount: {verts.shape[0]}")
    if verts.shape[0] > 10000:
        print(f"WARNING: vertices count over 10k ({verts.shape[0]}). Discarding mesh")
        m_valid = False
    elif (np.vstack(lbls) > 2).any():
        classes = [segment['name'] for segment in annot.segments]
        print(f"WARNING: has content segment in mesh ({classes}). Discarding mesh")
        m_valid = False

    ans = ''
    if not m_valid:
        ans = 'no'
    elif not manual:
        ans = 'yes'
    else:
        colors = np.vstack(colors)
        pc = trimesh.PointCloud(vertices=shapenet_mesh.vertices, colors=colors)
        pc.show()

    valid = {'yes': True, 'y': True, 'no': False, 'n': False}

    while ans not in valid.keys():
        ans = input("Mesh valid? [y/n]: ")
    if valid[ans]:
        labels = np.vstack(labels)
        # Save ShapeNet mesh with ParNet labels
        with open(f'{out_npz_dir}/{annot.anno_id}.npz', 'wb') as f:
            np.savez(f, verts=shapenet_mesh.vertices, faces=shapenet_mesh.faces, labels=labels)
