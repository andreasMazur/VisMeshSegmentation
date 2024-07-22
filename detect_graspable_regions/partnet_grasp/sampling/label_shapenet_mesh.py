from detect_graspable_regions.partnet_grasp.sampling.constants import CAT2SYNSET, LABLEMAP, COLORMAP, TMPMAP
from detect_graspable_regions.partnet_grasp.sampling.utils import ModelHandler

from io import BytesIO

import os
import os.path as osp
import trimesh
import open3d as o3d
import numpy as np
import zipfile
import pathlib


def orig_segmentation(
            aligned_archive: zipfile.ZipFile,
            model: ModelHandler,
            out_npz_dir: pathlib.Path | str,
            manual: bool = False
        ) -> None:
    """Computes aligned ShapeNet labels from signed distance to PartNet segments.

    Parameters
    ----------
    aligned_archive : zipfile.ZipFile
        zip handler to the aligned dataset.
    model: ModelHandler
        Handler to the mesh to process
    out_npz_dir : pathlib.Path | str
        Directory where to save a numpy file containing vertices, faces, and segment labels.
    manual : bool, optional
        If True, the user is prompted for each mesh to decide whether it should be included in the output dataset.
        If False, the same selection of the paper is used. By default False.

    Raises
    ------
    ValueError
        If the category is unknown
    """

    os.makedirs(out_npz_dir, exist_ok=True)

    cat_name = model.meta['model_cat'].lower()
    model_id = model.meta['model_id']

    if cat_name not in CAT2SYNSET.keys():
        raise ValueError(f"Category '{cat_name}' not in dictionary map to shapenet!")

    # Load segments as individual meshes

    model_aligned_path = osp.join(CAT2SYNSET[cat_name], model_id, 'training', 'model_normalized.obj')
    buf = BytesIO(aligned_archive.read(model_aligned_path))
    shapenet_mesh = trimesh.load_mesh(
        buf,
        maintain_order=False,
        skip_materials=True,
        process=False,
        file_type="obj"
    )
    if isinstance(shapenet_mesh, trimesh.Scene):
        shapenet_mesh = trimesh.util.concatenate(list(shapenet_mesh.geometry.values()))

    # sadly no cuda support for embree yet , device=o3d.core.Device("CUDA:0"))
    verts = o3d.core.Tensor(shapenet_mesh.vertices, dtype=o3d.core.Dtype.Float32)

    # assign a label to each ShapeNet model vertex, based on the closest vertex 
    # of the combined PartNet model it is aligned to.
    distances = []
    lbls = []
    for t_mesh, lbl in model.partnet_segments:
        o3d_mesh = t_mesh.as_open3d
        o3d_mesh = o3d.t.geometry.TriangleMesh.from_legacy(o3d_mesh)
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(o3d_mesh)
        dist = scene.compute_signed_distance(verts)
        distances.append(dist)
        lbls.append(lbl)

    # Get index i.e label with min dist
    res = np.stack(distances, axis=-1).argmin(axis=1)

    # colorize different segments
    labels = []
    colors = []
    for _, vert in enumerate(res):
        colors.append(COLORMAP[TMPMAP[lbls[vert]]])
        labels.append(lbls[vert]-1)

    # Check if mesh fulfills criteria
    m_valid = True
    print(f"INFO: vertcount: {verts.shape[0]}")
    if verts.shape[0] > 10000: # mesh should have no more than 10k vertices
        print(f"WARNING: vertices count over 10k ({verts.shape[0]}). Discarding mesh")
        m_valid = False
    elif (np.vstack(lbls) > 2).any(): # mesh should not containt contents 
        classes = [segment['name'] for segment in model.segments]
        print(f"WARNING: has content segment in mesh ({classes}). Discarding mesh")
        m_valid = False

    ans = ''
    # if mesh is invalid choose to discard automatically
    if not m_valid:
        ans = 'no'
    # if not manual this mesh already has been selected
    elif not manual:
        ans = 'yes'
    # if manual, show the mesh first
    else:
        colors = np.vstack(colors)
        pc = trimesh.PointCloud(vertices=shapenet_mesh.vertices, colors=colors)
        pc.show()

    valid = {'yes': True, 'y': True, 'no': False, 'n': False}

    # if manual, user is prompted to choose to discard or keep the mesh in the dataset
    while ans not in valid.keys():
        ans = input("Mesh valid? [y/n]: ")
    if valid[ans]:
        labels = np.vstack(labels)
        # Save ShapeNet mesh with ParNet labels
        with open(f'{out_npz_dir}/{model.anno_id}.npz', 'wb') as f:
            np.savez(f, verts=shapenet_mesh.vertices, faces=shapenet_mesh.faces, labels=labels)
