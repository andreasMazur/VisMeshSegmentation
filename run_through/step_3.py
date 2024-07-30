from improve_mesh_segmentation.partnet_grasp.preprocess import preprocess_data
from improve_mesh_segmentation.partnet_grasp.sampling.convert_partnet_labels import convert_partnet_labels
from improve_mesh_segmentation.partnet_grasp.sampling.utils import PartNetDataset, ShapeNetDataset

# TODO: Specify the path to this repository!
EXPERIMENT_DIRECTORY = "PATH/TO/VisMeshSegmentation"

DATASETS_PATH = f"{EXPERIMENT_DIRECTORY}/datasets"
SAMPLED_PARTNET = f"{DATASETS_PATH}/sampled_partnet"
PARTNET_GRASP = f"{DATASETS_PATH}/partnet_grasp"

if __name__ == "__main__":
    convert_partnet_labels(
        partnet_dataset=PartNetDataset(f"{DATASETS_PATH}/PartNet-archive"),
        shapenet_dataset=ShapeNetDataset(DATASETS_PATH),
        aligned_archive_path=f"{DATASETS_PATH}/aligned_shapenet",
        target_dataset_path=SAMPLED_PARTNET,
        obj_class="Mug",
        manual=False
    )
    preprocess_data(
        data_path=SAMPLED_PARTNET,
        target_dir=PARTNET_GRASP,  # will become a ZIP-file, i.e. 'partnet_grasp.zip'
        processes=10,  # Adjust to how many CPU-cores you wish to use for preprocess
        n_radial=5,
        n_angular=8
    )
