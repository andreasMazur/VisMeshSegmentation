import os
from collections import defaultdict
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from improve_mesh_segmentation.partnet_grasp.dataset import PartNetGraspDataset, processed_partnet_grasp_generator




og_data_path = "/home/iroberts/projects/VisMeshSegmentation/improve_mesh_segmentation/datasets/partnet_grasp.zip"
corrected_data_path = "/home/iroberts/projects/VisMeshSegmentation/improve_mesh_segmentation/datasets/partnet_grasp_corrected.zip"
# corrected_data_path = "/home/iroberts/projects/VisMeshSegmentation/improve_mesh_segmentation/datasets/partnet_grasp_corrected_deepviewbackground_100.zip"
# method1_path = "/improve_mesh_segmentation/datasets/deepview_influence_sort_percentage/partnet_grasp_corrected_deepview_influence_background_100.zip"
# method2_path = "/improve_mesh_segmentation/datasets/deepview_background_percentage/partnet_grasp_corrected_deepviewbackground_100.zip"
# method3_path = "/home/iroberts/projects/VisMeshSegmentation/improve_mesh_segmentation/datasets/partnet_grasp_corrected_deepview_preds_100.zip"
#
# unc_baseline = "/home/iroberts/projects/VisMeshSegmentation/improve_mesh_segmentation/datasets/misclassification_uncertainty_baseline_percentage/"
#
# inf_baseline = "/home/iroberts/projects/VisMeshSegmentation/improve_mesh_segmentation/datasets/misclassification_influence_baseline_percentage/"

# [ "global_knn_true"  + str(j) for j in [1000, 1500, 2000, 3000, 4000, 5000]] + ['global_lvq_4', 'global_knn_5000', 'misclassification_sorted_unc','misclassification_sorted_inf']
        # [ "global_knn_100", "global_knn_250", "global_knn_500",'misclassification_sorted_unc', 'misclassification_sorted_inf'
        #             , "global_knn_1000", "global_knn_1500", "global_knn_2000", "global_knn_3000", "global_knn_4000", "global_knn_5000", ]
                    # 'deepview_background_random', "deepview_background_sorted_inf", "deepview_background_sorted_unc",
                    #  "deepview_kmeans","global_kmeans","knn_100_corrections","global_dbscan"]
                    #"knn_30_corrections", "knn_5_corrections", "deepview_kmeans7","deepview_dbscan","global_knn_5", "global_knn_10", "global_knn_25", "global_knn_50",]

datasets_path = "/home/iroberts/projects/VisMeshSegmentation/improve_mesh_segmentation/datasets/"


if __name__ == "__main__":
    # Load shared datasets into memory (as lists)
    og_dataset = list(processed_partnet_grasp_generator(og_data_path, set_type=0))
    corrected_dataset = list(processed_partnet_grasp_generator(corrected_data_path, set_type=0))

    # List of automated correction methods and their datasets (also convert to list)
    automated_methods = {

    }
    # method_names =  ['iterative_kmeans','iterative_knn','deepview_kmeanstl','deepview_knn_bg','deepview_knn_t','confident_learning','cv_majority_baseline','global_lvq_4', 'global_knn_5000','global_knn_true500', 'misclassification_sorted_unc',]

    # dirs = ["global_knn_true" + str(j) for j in [5,10,25,50,100,250,500]]
    # dirs = ["global_lvq_" + str(j) for j in list(range(1,11))]
    dirs = ['random_baseline','iterative_kmeans','iterative_knn','iterative_knn1','deepview_kmeanstl','deepview_knn_bg','deepview_knn_t','confident_learning','cv_majority_baseline','global_lvq_4', 'global_knn_5000','global_knn_true500', 'misclassification_sorted_unc',]

    for dir in dirs:
        files = datasets_path + dir
        for file in os.listdir(files):
            automated_methods[str(file)] = list(processed_partnet_grasp_generator(files +"/" + file, set_type=0))

    # for file in os.listdir(inf_baseline):
    #     automated_methods[str(file)] = list(processed_partnet_grasp_generator(inf_baseline + file, set_type=0))

    # key = method name, value = dict of lists for precision/recall/F1 of corrections
    correction_agreement = defaultdict(lambda: {"precision": [], "recall": [], "f1": [], "no_of_changes":[],"oracle_changes":[]})

    # Loop through each method
    for method_name, auto_dataset in automated_methods.items():
        print(f"Evaluating correction overlap for {method_name}...")

        for mesh_idx, (((_, _), og_labels), ((_, _), cor_labels), ((_, _), auto_labels)) in enumerate(
                zip(og_dataset, corrected_dataset, auto_dataset)
        ):
            og_labels = np.array(og_labels)
            cor_labels = np.array(cor_labels)
            auto_labels = np.array(auto_labels)

            if not (og_labels.shape == cor_labels.shape == auto_labels.shape):
                print(f"[{method_name}] Shape mismatch at mesh {mesh_idx}")
                continue

            # Masks where the label was changed
            oracle_changed = og_labels != cor_labels
            auto_changed = og_labels != auto_labels

            # Binary labels: 1 if correction, 0 if no change
            y_true = oracle_changed.astype(int)
            y_pred = auto_changed.astype(int)
            # print("method name:", sum(y_true), sum(y_pred))

            # Metrics: did the automated method change the *same* labels?
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            # print(classification_report(y_true,y_pred))
            correction_agreement[method_name]["precision"].append(prec)
            correction_agreement[method_name]["recall"].append(rec)
            correction_agreement[method_name]["f1"].append(f1)
            correction_agreement[method_name]["no_of_changes"].append(sum(y_pred))
            correction_agreement[method_name]["oracle_changes"].append(sum(y_true))


    # Print summary
    for method_name, metrics in correction_agreement.items():
        print(f"\n=== Correction Agreement: {method_name} ===")
        for metric, values in metrics.items():
            if metric != "no_of_changes":
                print(f"{metric.capitalize()}: Mean = {np.mean(values):.3f}, Std = {np.std(values):.3f}")
            if metric == "oracle_changes":
                print(f"{metric.capitalize()}: sum = {np.sum(values):.3f}")
            else:
                print(f"{metric.capitalize()}: sum = {np.sum(values):.3f}")