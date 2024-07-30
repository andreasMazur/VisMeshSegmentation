# Visualizing and Improving 3D Mesh Segmentation with DeepView

This repository contains the **code** and an **experiment protocol** to replicate the results published in:

&emsp; **Visualizing and Improving 3D Mesh Segmentation with DeepView**

## Step 1: Technical prerequisites

This experiment builds mainly on the following code:
- [DeepView](https://github.com/LucaHermes/DeepView) - A framework for visualizing classification functions of deep
  neural networks.
- [GeoConv](https://github.com/andreasMazur/geoconv) - A library for coding Intrinsic Mesh CNNs.
  - In this experiment we use GeoConv in combination with [Pytorch](https://pytorch.org/).

We suggest to setup a local [Conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)-environment
using **Python 3.10** and install the repository as follows:

```bash
git clone https://github.com/andreasMazur/VisMeshSegmentation.git
pip install -r requirements.txt
pip install pyshot@git+https://github.com/uhlmanngroup/pyshot@master
pip install deepview@git+https://github.com/LucaHermes/DeepView@master
```

In case OpenGL context cannot be created:
```bash
conda install -c conda-forge libstdcxx-ng
```

## Step 2: Download PartNet and ShapeNet

Within the experiment portions of the PartNet- and ShapeNet-datasets are used. More details on the datasets can be found
on the corresponding dataset websites:
- [ShapeNet](https://shapenet.org/) - An Information-Rich 3D Model Repository.
- [PartNet](https://partnet.cs.stanford.edu/) - A Large-scale Benchmark for Fine-grained and Hierarchical Part-level 3D Object
  Understanding

The datasets must be downloaded from [Hugging Face](https://huggingface.co):
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

Create a `datasets` folder and move both datasets into it:
```bash
mkdir -p ~/datasets/ShapeNetCore.v2
mkdir ~/datasets
mv PATH/TO/DOWNLOADED/ShapeNetCore/03797390.zip ~/datasets/ShapeNetCore.v2/03797390.zip
mv PATH/TO/DOWNLOADED/PartNet-archive ~/datasets/PartNet-archive
```

Remember your `datasets` path:
```python
DATASETS_PATH = "ABSOLUTE/PATH/TO/datasets"
```

## Step 3: Sampling and preprocessing PartNet-Grasp

In this experiment, we work with the *'Mug'*-class of ShapeNet and PartNet, respectively.
Sample, align the PartNet-meshes to the ShapeNet-meshes to get segmentation labels for ShapeNet-meshes and
preprocess them for subsequently training an *Intrinsic Mesh CNN* (IMCNN) as follows:

First define the path for where the **sampled and aligned shapes** shall be stored. E.g:
```python
SAMPLED_PARTNET = f"{DATASETS_PATH}/SampledPartNet"
```

Start the sampling and alignment process by executing the following script:

```python
from improve_mesh_segmentation.partnet_grasp.sampling.convert_partnet_labels import convert_partnet_labels
from improve_mesh_segmentation.partnet_grasp.sampling.utils import PartNetDataset, ShapeNetDataset

if __name__ == "__main__":
    convert_partnet_labels(
        partnet_dataset=PartNetDataset(DATASETS_PATH),
        shapenet_dataset=ShapeNetDataset(DATASETS_PATH),
        aligned_archive_path=f"{DATASETS_PATH}/AlignedShapeNet",
        target_dataset_path=SAMPLED_PARTNET,
        obj_class="Mug",
        manual=False
    )
```

Afterwards, define the path to where we can store preprocessing results for the subsequent training of 
IMCNNs. E.g:
```python
PARTNET_GRASP = f"{DATASETS_PATH}/PartNetGrasp"
```

The preprocess is then started by running:

```python
from improve_mesh_segmentation.partnet_grasp.preprocess import preprocess_data

if __name__ == "__main__":
    preprocess_data(
        data_path=SAMPLED_PARTNET,
        target_dir=PARTNET_GRASP,  # will become a ZIP-file, i.e. 'PartNetGrasp.zip'
        processes=10,  # Adjust to how many CPU-cores you wish to use for preprocess
        n_radial=5,
        n_angular=8
    )
```

## Step 4: Train an initial IMCNN

A trained IMCNN is required for the label correction process. Define the path to a training logs directory:
```python
TRAINING_LOGS = "PATH/TO/TRAINING/LOGS"
```

Train an IMCNN by executing the following script:
```python
from improve_mesh_segmentation.training.train_imcnn import train_single_imcnn


if __name__ == "__main__":
    train_single_imcnn(
        data_path=PARTNET_GRASP,
        n_epochs=10,
        logging_dir=TRAINING_LOGS,
        skip_validation=False,
        skip_testing=False,
        verbose=True
    )
```

## Step 5: Correcting Segmentation Labels

Now the noisy segmentation labels from the alignment process need to be corrected.
There are two options for this.
One can either use our pre-made corrections by referring to the correction file within this repository:
```python
CORRECTIONS_FILE = "PATH/TO/improve_mesh_segmentation/data_correction/partnet_correction.csv"
```
Alternatively, one can run the segmentation label correction algorithm to interactively
correct segmentation labels of PartNet-Grasp.
For this, define the path to where the corrections-file shall be stored:

```python
CORRECTIONS_FILE = "/PATH_TO/partnet_correction.csv"
```

Then, execute the following script to start the correction algorithm:
```python
from improve_mesh_segmentation.data_correction.correct_sub_partnet import correct_sub_partnet

import matplotlib

if __name__ == "__main__":
    matplotlib.use("Qt5Agg")
    correct_sub_partnet(
        data_path=PARTNET_GRASP,
        model_path=f"{TRAINING_LOGS}/model.zip",
        correction_csv_path=CORRECTIONS_FILE
    )
```

## Step 6: Including the label corrections into the dataset

After correcting, the corrected labels need to be incorporated into the originally
preprocessed dataset.
For this two new paths need to be defined:
1. Corrected dataset shall be stored in: `PARTNET_GRASP_CORRECTED = "PATH/TO/corrected_PartNetGrasp"`
2. Label change arrays shall be stored in: `LABEL_CHANGES = "PATH/TO/label_changes"`

```python
from improve_mesh_segmentation.data_correction.convert_partnet import convert_partnet

if __name__ == "__main__":
    convert_partnet(
        old_data_path=PARTNET_GRASP,
        new_data_path=PARTNET_GRASP_CORRECTED,
        csv_path=CORRECTIONS_FILE,
        label_changes_path=LABEL_CHANGES  # Required for 'Step 8'
    )
```

## Step 7: Hypothesis Test

Now the hypothesis test from the paper can be run by exectuing:

```python
from improve_mesh_segmentation.experiments.hypothesis_test import run_hypothesis_test

if __name__ == "__main__":
    run_hypothesis_test(
        old_dataset_path=PARTNET_GRASP,
        new_dataset_path=PARTNET_GRASP_CORRECTED,
        csv_path=CORRECTIONS_FILE,
        logging_dir="./hypothesis_test_logs",
        trials=30,
        epochs=10
    )
```

## Step 8: Comparison to Filter Method

Eventually, the results for the comparison to the filter method can be run by executing:

```python
from improve_mesh_segmentation.experiments.cross_validation import (
    partnet_grasp_cross_validation, filter_method
)

if __name__ == "__main__":
    cv_logs = "./cross_validation_logs"
    partnet_grasp_cross_validation(
        k=5,  # 5-fold cross validation
        epochs=10,
        zip_file=PARTNET_GRASP,  # uncorrected dataset
        logging_dir=cv_logs,
        label_changes_path=LABEL_CHANGES
    )
    filter_method(cv_logs)
```
