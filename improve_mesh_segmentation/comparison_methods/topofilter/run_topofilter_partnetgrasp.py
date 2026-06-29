from pathlib import Path

from train_pointnet import train_models as pointnet_train
from train_imcnn import train_models as imcnn_train

def run_topofilter_experiments(
    model_type : str,
    partnet_grasp_path : Path,
    correction_file_path : Path,
    unq_v_triplets_path : Path,
    results_path : Path,
    k_ccs,
    start_cleans,
    everys,
    zetas,
    k_outliers,
):
    if model_type == "pointnet" or model_type == "all":
        pointnet_train(
            partnet_grasp_path,
            correction_file_path,
            unq_v_triplets_path,
            results_path,
            k_ccs=k_ccs,
            start_cleans=start_cleans,
            everys=everys,
            zetas=zetas,
            k_outliers=k_outliers
        )
    if model_type == "imcnn" or model_type == "all":
        imcnn_train(
            partnet_grasp_path,
            correction_file_path,
            unq_v_triplets_path,
            results_path,
            k_ccs=k_ccs,
            start_cleans=start_cleans,
            everys=everys,
            zetas=zetas,
            k_outliers=k_outliers
        )