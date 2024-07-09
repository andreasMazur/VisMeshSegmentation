from detect_graspable_regions.partnet_grasp.sampling.alignment import align_to_partnet
from detect_graspable_regions.partnet_grasp.sampling.constants import PARTNET_EXP_GITHUB_LINK
from detect_graspable_regions.partnet_grasp.sampling.label_shapenet_mesh import orig_segmentation
from detect_graspable_regions.partnet_grasp.sampling.mesh_filtering import manual_filter_mesh
from detect_graspable_regions.partnet_grasp.sampling.utils import ModelHandler, PartNetDataset, ShapeNetDataset
from detect_graspable_regions.partnet_grasp.sampling.partnet_grasp_meshes import ANNOT_DICT

import json
import urllib.request

import zipfile


def convert_partnet_labels(
        partnet_dataset: PartNetDataset,
        shapenet_dataset: ShapeNetDataset,
        aligned_archive_path: str,
        target_dataset_path: str,
        obj_class: str = 'Mug',
        manual: bool = False) -> None:
    """Transfers PartNet labels to ShapeNet mesh.

    Involves
    a) extracting ShapeNet meshes of the given class from ShapeNet,
    b) transforming the ShapeNet mesh to match the combined PartNet transform,
    c) assigning labels to the transformed ShapeNet mesh by choosing the min distance of signed distance with each
     individual segment.
    d) saving vertices, faces, and vertex labels as npz files under `target_dataset_path`.

    Parameters
    ----------
    partnet_dataset : str
        Handler to the PartNet data set.
    shapenet_dataset : str
        Handler to the ShapeNet data set.
    aligned_archive_path : str
        Destination archive to save the aligned ShapeNet in.
    target_dataset_path : str
        Destination dir to save the extracted dataset to.
    obj_class : str, optional
        Object class name, by default 'Mug'
    manual : bool, optional
        Whether to manually select meshes. Defaults to False.
    """

    # Truncate existing target file and start new
    target_zip = zipfile.ZipFile(aligned_archive_path, 'w', zipfile.ZIP_DEFLATED)


    # Load annotation data from official PartNet-repository
    data = []
    for split in ['train', 'test', 'val']:
        with urllib.request.urlopen(f"{PARTNET_EXP_GITHUB_LINK}/{obj_class}.{split}.json") as url:
            d = json.load(url)
            data.extend(d)

    for inst in data:
        # Create annotation object
        model = ModelHandler(
            anno_id=inst['anno_id'],
            partnet_dataset=partnet_dataset,
            shapenet_dataset=shapenet_dataset
        )

        # Align PartNet-segment-meshes to corresponding ShapeNet-mesh
        align_to_partnet(
            aligned_archive=target_zip,
            model=model,
        )

        # Select or dismiss aligned mesh (manually or according to 'ANNOT_DICT')
        if manual:
            use_mesh = manual_filter_mesh(
                model=model,
            )
        else:
            use_mesh = ANNOT_DICT[inst['anno_id']]

        # Save labeled ShapeNet-mesh
        if use_mesh:
            orig_segmentation(
                model=model,
                aligned_archive=target_zip,
                out_npz_dir=target_dataset_path,
                manual=manual
            )
