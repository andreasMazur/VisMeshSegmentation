from improve_mesh_segmentation.experiments.cross_validation import partnet_grasp_cross_validation, filter_method
from run_through.step_3 import PARTNET_GRASP
from run_through.step_4 import LOGGING_DIR
from run_through.step_6 import LABEL_CHANGES


if __name__ == "__main__":
    cv_logs = f"{LOGGING_DIR}/cross_validation_logs"
    partnet_grasp_cross_validation(
        k=5,  # 5-fold cross validation
        epochs=10,
        zip_file=f"{PARTNET_GRASP}.zip",  # uncorrected dataset
        logging_dir=cv_logs,
        label_changes_path=LABEL_CHANGES
    )
    filter_method(cv_logs)
