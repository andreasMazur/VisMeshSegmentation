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

Create a folder called `datasets` **within this repository directory (!)** and move both datasets into it:
```bash
mkdir -p PATH/TO/VisMeshSegmentation/datasets/ShapeNetCore.v2
mv PATH/TO/DOWNLOADED/ShapeNetCore/03797390.zip PATH/TO/VisMeshSegmentation/datasets/ShapeNetCore.v2/03797390.zip
mv PATH/TO/DOWNLOADED/PartNet-archive PATH/TO/VisMeshSegmentation/datasets/PartNet-archive
```

## Step 3 to 8: Repeat the experiments from the paper

In order to replicate the experiments from the paper, head into the `run_through` folder and execute scripts `step_3.py`
to `step_8.py` in order.
Thereby, carefully read the instructions given in the doc-strings within the respective script-files.
