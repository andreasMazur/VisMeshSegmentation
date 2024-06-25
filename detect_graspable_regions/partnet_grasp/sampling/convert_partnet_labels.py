from detect_graspable_regions.partnet_grasp.sampling.alignment import align_to_partnet
from detect_graspable_regions.partnet_grasp.sampling.constants import PARTNET_EXP_GITHUB_LINK
from detect_graspable_regions.partnet_grasp.sampling.label_shapenet_mesh import orig_segmentation
from detect_graspable_regions.partnet_grasp.sampling.mesh_filtering import manual_filter_mesh
from detect_graspable_regions.partnet_grasp.sampling.partnet_annotated_mesh import PartNetAnnotatedMesh
from detect_graspable_regions.partnet_grasp.sampling.partnet_grasp_meshes import ANNOT_DICT

import json
import urllib.request


def convert_partnet_labels(
        base_partnet_path: str,
        base_shapenet_path: str,
        target_mesh_path: str,
        target_dataset_path: str,
        obj_class: str = 'Mug',
        manual: bool = False,
        verbose: bool = False) -> None:
    """Transfers PartNet labels to ShapeNet mesh.
    Involves
    a) extracting ShapeNet meshes of the given class from ShapeNet,
    b) transforming the ShapeNet mesh to match the combined PartNet transform,
    c) assigning labels to the transformed ShapeNet mesh by choosing the min distance of signed distance with each
     individual segment.
    d) saving vertices, faces, and vertex labels as npz files under `target_dataset_path`.

    Args:
        base_partnet_path (str): Path to `PartNet/data_vX` dir.
        base_shapenet_path (str): Path to `ShapeNetCore.vX` dir.
        target_mesh_path (str): Destination dir to save the aligned ShapeNet partnet_grasp to.
        target_dataset_path (str): Destination dir to save the extracted dataset to.
        obj_class (str, optional): Object class name. Defaults to 'Mug'.
        manual (bool, optional): Whether to manually select meshes. Defaults to False.
        verbose (bool, optional): Show more info. Defaults to False.
    """

    # Load annotation data from official PartNet-repository
    data = []
    for split in ['train', 'test', 'val']:
        with urllib.request.urlopen(f"{PARTNET_EXP_GITHUB_LINK}/{obj_class}.{split}.json") as url:
            d = json.load(url)
            data.extend(d)

    for inst in data:
        # Create annotation object
        annot = PartNetAnnotatedMesh(
            anno_id=inst['anno_id'],
            base_partnet_path=base_partnet_path,
            base_shapenet_path=base_shapenet_path
        )

        # Align PartNet-segment-meshes to corresponding ShapeNet-mesh
        align_to_partnet(
            base_partnet_path=base_partnet_path,
            base_shapenet_path=base_shapenet_path,
            aligned_shapenet_target_path=target_mesh_path,
            anno_id=annot.anno_id,
            cat_name=annot.meta['model_cat'].lower(),
            model_id=annot.meta['model_id'],
            verbose=verbose
        )

        # Select or dismiss aligned mesh (manually or according to 'ANNOT_DICT')
        if manual:
            use_mesh = manual_filter_mesh(
                base_partnet_path=base_partnet_path,
                aligned_shapenet_path=target_mesh_path,
                anno_id=annot.meta['anno_id'],
                cat_name=annot.meta['model_cat'].lower(),
                model_id=annot.meta['model_id']
            )
        else:
            use_mesh = ANNOT_DICT[inst['anno_id']]

        # Save labeled ShapeNet-mesh
        if use_mesh:
            orig_segmentation(
                base_partnet_path=base_partnet_path,
                aligned_shapenet_path=target_mesh_path,
                annot=annot,
                out_npz_dir=target_dataset_path,
                verbose=verbose,
                manual=manual
            )
