# Visualizing and Improving 3D Mesh Segmentation with DeepView

## Technical prerequisites

In order to run this example, further dependencies need to be installed:

```bash
pip install -r requirements.txt
cd PATH/TO/geoconv
pip install .
pip install cython==0.29.37
pip install pyshot@git+https://github.com/uhlmanngroup/pyshot@master
pip install torch
pip install deepview@git+https://github.com/LucaHermes/DeepView@master
```

In case OpenGL context cannot be created:
```bash
conda install -c conda-forge libstdcxx-ng
```

## Download PartNet and ShapeNet

```python
from huggingface_hub import login

import datasets

if __name__ == "__main__":
    """Download Mug-meshes from ShapeNetCore.v2 and PartNet-archive
    
    You can also do it manually from here:
    * ShapeNetCore.v2:
        https://huggingface.co/datasets/ShapeNet/ShapeNetCore/blob/main/03797390.zip
    * PartNet-archive:
        https://huggingface.co/datasets/ShapeNet/PartNet-archive/tree/main
    """
    login()
    datasets.load_dataset("ShapeNet/ShapeNetCore", data_files="03797390.zip")
    datasets.load_dataset("ShapeNet/PartNet-archive")
```

The PartNet dataset requires an extra installation step as it comes as a split dataset.
Head to the download folder that contains the `data_v0_chunk.z*`-files and execute the following command to combine the splitted dataset into one zip-file:

```bash
zip -s 0 data_v0_chunk.z* --out PartNetData.zip
```

Remember your paths to the folder which contain the downloaded datasets:
```python
SHAPENET_DS_PATH = "PATH/TO/ShapeNetCore.v2"  # Folder that contains the synset zip-files
PARTNET_DS_PATH = "PATH/TO/PartNet-archive"  # Folder that contains 'PartNetData.zip'
```

## Sampling and preprocessing PartNet-Grasp

Now we want to sample, align and preprocess PartNet-Grasp.

Define the path for where the sampled and aligned shapes of PartNet-grasp shall be stored:
```python
SAMPLED_PARTNET = "PATH/TO/SampledPartNet"
```

Sample and align the *'Mug'*-class of PartNet by executing the following script:
```python
from detect_graspable_regions.partnet_grasp.sampling.convert_partnet_labels import convert_partnet_labels
from detect_graspable_regions.partnet_grasp.sampling.utils import PartNetDataset, ShapeNetDataset

if __name__ == '__main__':
    convert_partnet_labels(
        partnet_dataset=PartNetDataset(PARTNET_DS_PATH),
        shapenet_dataset=ShapeNetDataset(SHAPENET_DS_PATH),
        aligned_archive_path="PATH/TO/AlignedShapeNet",
        target_dataset_path=SAMPLED_PARTNET,
        obj_class="Mug",
        manual=False
    )
```

Subsequently, define the path to another directory:
```python
PARTNET_GRASP_DS_PATH = "PATH/TO/partnet_grasp"
```
We use that directory to store preprocessed shapes for the subsequent training of IMCNNs.
The preprocess is started by running:
```python
from detect_graspable_regions.partnet_grasp.preprocess import preprocess_data

if __name__ == '__main__':
    preprocess_data(
        data_path=SAMPLED_PARTNET,
        target_dir=PARTNET_GRASP_DS_PATH,  # will become a ZIP-file, i.e. 'partnet_grasp.zip'
        processes=10,  # Adjust to how many CPU-cores you wish to use for preprocess
        n_radial=5,
        n_angular=8
    )
```

## Correcting Segmentation Labels

Next, the user needs to run the segmentation label correction algorithm to interactively
correct segmentation labels of PartNet-Grasp:

```python
from geoconv_examples.detect_graspable_regions.data_correction.correct_sub_partnet import correct_sub_partnet

if __name__ == "__main__":
    correct_sub_partnet(
        data_path="/PATH_TO/partnet_grasp.zip",
        model_path="/PATH_TO/saved_imcnn.zip",
        correction_csv_path="/PATH_TO/partnet_proposed_corrections.csv"
    )
```

After correcting the labels the user needs to incorporate the changes into the originally
preprocessed dataset:

```python
from geoconv_examples.detect_graspable_regions.data_correction.convert_partnet import convert_partnet

if __name__ == "__main__":
    convert_partnet(
        old_data_path="/PATH_TO/partnet_grasp",  # preprocessed original dataset (uncorrected)
        new_data_path="/PATH_TO/partnet_grasp_deepview",  # corrected dataset
        csv_path="/PATH_TO/partnet_proposed_corrections.csv",  # csv containing corrections
        label_changes_path="/PATH_TO/label_changes"  # binary array, telling whether label of vertex 'i' has changed
    )
```

## Hypothesis Test

Run the hypothesis test. If proposed label corrections are not already incorporated into the dataset,
this function will do it. Otherwise, corrected data will be loaded if path is set correctly.

```python
from geoconv_examples.detect_graspable_regions.experiments.hypothesis_test import run_hypothesis_test

if __name__ == "__main__":
    run_hypothesis_test(
        old_dataset_path="/PATH_TO/partnet_grasp",
        new_dataset_path="/PATH_TO/partnet_grasp_deepview",
        csv_path="/PATH_TO/partnet_proposed_corrections.csv",
        logging_dir="./hypothesis_test_logs",
        trials=30,
        epochs=10
    )
```

## Comparison to Filter Method

```python
from geoconv_examples.detect_graspable_regions.experiments.cross_validation import (
    partnet_grasp_cross_validation, filter_method
)

if __name__ == "__main__":
    partnet_grasp_cross_validation(
        k=5,  # 5-fold cross validation
        epochs=10,
        zip_file="/PATH_TO/partnet_grasp",  # uncorrected dataset
        logging_dir="./cross_validation_logs",
        label_changes_path="/PATH_TO/label_changes"
    )
    filter_method("./cross_validation_logs")
```