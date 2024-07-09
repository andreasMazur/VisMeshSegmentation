from detect_graspable_regions.partnet_grasp.sampling.utils import get_chamfer_dist, ModelHandler
from detect_graspable_regions.partnet_grasp.sampling.constants import CAT2SYNSET

import zipfile
import trimesh
import numpy as np


def manual_filter_mesh(model : ModelHandler, target_archive : zipfile.ZipFile) -> bool:
    """Dispaly the transformed and the target mesh and prompt the user if the mesh should be kept in the dataset.

    Parameters
    ----------
    model : ModelHandler
        Handler to the models in the respective datasets.
    target_archive : zipfile.ZipFile
        Zip archive the transformed mesh was saved to.

    Returns
    -------
    bool
        whether the user chose to keep the transformed mesh.

    Raises
    ------
    ValueError
        If the category is unknown.
    """

    cat_name=model.meta['model_cat'].lower(),
    model_id=model.meta['model_id']

    if cat_name not in CAT2SYNSET.keys():
        raise ValueError(f"Category '{cat_name}' not in dictionary map to shapenet!")

    # Combine PartNet segments in one mesh
    segments = []
    for seg, _ in model.partnet_segments:
        segments.append(seg)
    partnet_full_mesh = trimesh.util.concatenate(
        segments
    )

    # Load aligned ShapeNet mesh
    buf = target_archive.read(f"{CAT2SYNSET.get(cat_name)}/{model_id}/training/model_normalized.obj")
    shapenet_mesh = trimesh.load_mesh(
        buf,
        maintain_order=False,
        skip_materials=True,
        process=False,
        file_type="obj"
    )

    # Test chamfer distance
    chamfer_dist = get_chamfer_dist(shapenet_mesh, partnet_full_mesh)
    print(f"Chamfer Distance: {chamfer_dist}")

    # Color PartNet vertices and ShapeNet vertices differently
    # this allows for visual inspection of the alignment quality
    mesh = trimesh.util.concatenate([partnet_full_mesh, shapenet_mesh])
    colors = np.concatenate(
        [
            np.repeat([[0, 255, 0, 255]], len(partnet_full_mesh.vertices)).reshape(4, -1).T,
            np.repeat([[255, 127, 0, 255]], len(shapenet_mesh.vertices)).reshape(4, -1).T
        ]
    )
    pc = trimesh.PointCloud(vertices=mesh.vertices, colors=colors)
    pc.show()

    # promt user to select or discard this aligned mesh for the dataset
    ans = ''
    valid = {'yes': True, 'y': True, 'no': False, 'n': False}
    while ans not in valid.keys():
        ans = input("Use mesh? [y/n]: ")
    return valid[ans]
