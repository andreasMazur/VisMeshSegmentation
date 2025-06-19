

from improve_mesh_segmentation.partnet_grasp.dataset import PartNetGraspDataset, processed_partnet_grasp_generator

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score
import seaborn as sns



og_data_path = "/home/iroberts/projects/VisMeshSegmentation/improve_mesh_segmentation/datasets/partnet_grasp.zip"
corrected_data_path = "/home/iroberts/projects/VisMeshSegmentation/improve_mesh_segmentation/datasets/partnet_grasp_corrected.zip"


if __name__ == "__main__":
    # Set seaborn style for better visuals
    sns.set(style="whitegrid", palette="muted", font_scale=1.2)

    # --- Load datasets ---
    og_dataset = list(processed_partnet_grasp_generator(og_data_path, set_type=0))
    corrected_dataset = list(processed_partnet_grasp_generator(corrected_data_path, set_type=0))

    # Flatten original and human-corrected labels
    noisy_labels = []
    human_corrected_labels = []
    for ((_, _), og_labels), ((_, _), cor_labels) in zip(og_dataset, corrected_dataset):
        noisy_labels.extend(og_labels)
        human_corrected_labels.extend(cor_labels)

    og_labels = np.array(noisy_labels)
    cor_labels = np.array(human_corrected_labels)

    # --- Load automated correction datasets ---
    automated_methods = {}
    ks = np.arange(500, 9500, 500)
    method_names =  ['iterative_kmeans','iterative_knn1','deepview_kmeanstl','deepview_dbscan','deepview_knn_bg','deepview_knn_t','confident_learning','cv_majority_baseline','global_lvq_4', 'global_knn_5000','global_knn_true500', 'misclassification_sorted_unc',]
        # [ "global_knn_100", "global_knn_250", "global_knn_500",'misclassification_sorted_unc', 'misclassification_sorted_inf'
        #             , "global_knn_1000", "global_knn_1500", "global_knn_2000", "global_knn_3000", "global_knn_4000", "global_knn_5000", ]
                    # 'deepview_background_random', "deepview_background_sorted_inf", "deepview_background_sorted_unc",
                    #  "deepview_kmeans","global_kmeans","knn_100_corrections","global_dbscan"]
                    #"knn_30_corrections", "knn_5_corrections", "deepview_kmeans7","deepview_dbscan","global_knn_5", "global_knn_10", "global_knn_25", "global_knn_50",]

    for method in method_names:
        for k in ks:
            method_path = f"/home/iroberts/projects/VisMeshSegmentation/improve_mesh_segmentation/datasets/{method}/{method}_k_{k}.zip"
            key = f"{method}_k_{k}"
            automated_methods[key] = list(processed_partnet_grasp_generator(method_path, set_type=0))

    # --- Initialize metric storage ---
    correction_agreement = defaultdict(lambda: {
        "ks": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "no_of_changes": []
    })

    # --- Evaluate all automated methods ---
    for method_name, auto_dataset in automated_methods.items():
        print(f"Evaluating correction overlap for {method_name}...")

        method_base, k_str = method_name.rsplit("_k_", 1)
        k_val = int(k_str)

        automated_corrected_labels = []
        for (_, _), auto_labels in auto_dataset:
            automated_corrected_labels.extend(auto_labels)

        auto_labels = np.array(automated_corrected_labels)

        # Binary masks for which labels were changed
        oracle_changed = og_labels != cor_labels
        auto_changed = og_labels != auto_labels

        y_true = oracle_changed.astype(int)
        y_pred = auto_changed.astype(int)

        # Compute metrics
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # Store results
        correction_agreement[method_base]["ks"].append(k_val)
        correction_agreement[method_base]["precision"].append(prec)
        correction_agreement[method_base]["recall"].append(rec)
        correction_agreement[method_base]["f1"].append(f1)
        correction_agreement[method_base]["no_of_changes"].append(sum(y_pred))


    # --- Plotting function ---
    def plot_metric(metric_name, correction_agreement):
        plt.figure(figsize=(10, 6))
        for method, results in correction_agreement.items():
            # Sort by k for proper plotting
            sorted_data = sorted(zip(results["ks"], results[metric_name]))
            ks, metrics = zip(*sorted_data)
            plt.plot(ks, metrics, label=method, marker='o', linewidth=2)

        plt.title(f'{metric_name.capitalize()} vs. Number of Corrections (k)', fontsize=14)
        plt.xlabel('k (Number of Corrections)', fontsize=12)
        plt.ylabel(metric_name.capitalize(), fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(title="Method", fontsize=10)
        plt.tight_layout()
        # plt.show()


    # --- Plot all metrics ---
    for metric in ["precision", "recall", "f1"]:
        plot_metric(metric, correction_agreement)
        plt.savefig("./performance_figs/"+str(metric))