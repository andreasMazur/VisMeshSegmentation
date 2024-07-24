from detect_graspable_regions.partnet_grasp.sampling.constants import CAT2SYNSET, LABLEMAP

from scipy.spatial.distance import cdist
from io import BytesIO
from typing import Generator

import numpy as np
import trimesh
import pathlib
import json
import zipfile


def get_chamfer_dist(m1 : trimesh.Trimesh, m2 : trimesh.Trimesh, samples=2000) -> float:
    """Computes Chamfer distance between two meshes.

    Parameters
    ----------
    m1 : trimesh.Trimesh
        First mesh to compare.
    m2 : trimesh.Trimesh
        Second mesh to compare.
    samples : int, optional
        number of samples to generate on the mesh surfaces, by default 2000

    Returns
    -------
    float
        Chamfer distance.
    """
    pc1 = trimesh.sample.sample_surface(m1, 2000)[0]
    pc2 = trimesh.sample.sample_surface(m2, 2000)[0]
    dist_mat = cdist(pc1, pc2)
    return dist_mat.min(0).mean() + dist_mat.min(1).mean()


class PartNetDataset():
    """Helper class for PartNet dataset"""
    def __init__(self, base_path):

        self._buf_map = {} 

        self.base_path = base_path = pathlib.Path(base_path)
        print(f"Looking for PartNet in {base_path}")

        # Search for PartNet zip
        partnet_zip = list(base_path.glob('PartNetData.zip'))

        if len(partnet_zip) != 0:
            print("Found PartNet zip archive")
            self.is_zip = True
            # Read PartNet zipfile
            print("Indexing PartNet zip archive. This may take some time...")
            self.archive = zipfile.ZipFile(partnet_zip[-1], 'r')
            print("Done indexing archive.")
            self.data_path = pathlib.Path(pathlib.Path(self.archive.filelist[0].filename).parts[0])
            return

        # Search for PartNet data dir
        unpacked_dirs = list(base_path.glob('data.v*/'))

        assert len(unpacked_dirs) > 0 , "Neither PartNet data dir nor PartNet zipfile could be found!"
        print("Found unpacked PartNet data dir")
        self.is_zip = False
        self.data_path = unpacked_dirs[-1]

    def get_meta(self, anno_id : str) -> tuple[dict, dict]:
        """Fetch meta information from json files of a given PartNet model.

        Parameters
        ----------
        anno_id : str
            annotation id of the model in PartNet.

        Returns
        -------
        tuple[dict, dict]
            segments and model metadata.
        """        

        segments, meta = None, None
        if self.is_zip:
            # is zip, open archive and extract bytes data to json
            with self.archive.open(str(self.data_path/anno_id/"result.json")) as bs:
                result = json.loads(bs.read())[0]
                segments = result['children']
            with self.archive.open(str(self.data_path/anno_id/"meta.json")) as bs:
                meta = json.loads(bs.read())
            return segments, meta

        # is not zip, open file path and read json
        with open(self.data_path/anno_id/"result.json") as f:
            result = json.load(f)[0]
            segments = result['children']
        with open(self.data_path/anno_id/"meta.json") as bs:
            meta = json.load(bs)
        return segments, meta
        
    def open_model(self, model_path: str) -> trimesh.Trimesh:
        """Load PartNet model as trimesh model.

        Parameters
        ----------
        model_path : str
            relative path to model in dataset.

        Returns
        -------
        trimesh.Trimesh
            mesh loaded in trimesh without extra processing.
        """
        buf = self.get_model_buf(model_path)
        return trimesh.load_mesh(
            buf,
            maintain_order=False,
            skip_materials=True,
            process=False,
            file_type="obj"
        )

    def get_segment_paths(self, anno_id : str) -> pathlib.Path:
        """Retrieve segment paths of a PartNet model.

        Parameters
        ----------
        anno_id : str
            annotation id of the model in PartNet.

        Returns
        -------
        pathlib.Path
            If the dataset is loaded as zip, the path is relative to the archive root, otherwise it's absolute.
            returned paths should be opened with this class' `open_model` helper function.
        """

        objs_path = self.data_path / anno_id / 'objs'
        if self.is_zip:
            # Get files that are in the correct path within the zip archive
            return [p for p in self.archive.namelist() if str(objs_path) in p and p.endswith('.obj')]
        return [p.filename for p in objs_path.glob('*.obj')]

    def get_model_buf(self, model : pathlib.Path | str) -> BytesIO:
        """Read model contents into memory buffer.

        this method automatically resolves the correct path, whether it is read from an archive or an extracted data set.

        Parameters
        ----------
        model : pathlib.Path | str
            path to model file

        Returns
        -------
        BytesIO
            model file content in memory buffer
        """
        # Check if file is already in cache
        buf = self._buf_map.get(model)
        if buf is not None:
            # set to initial position such that read works
            buf.seek(0)
            return buf

        if self.is_zip:
            with self.archive.open(model, 'r') as fs:
                buf = BytesIO(fs.read())
        else:
            with open(model, 'rb') as fs:
                buf = BytesIO(fs.read())
        self._buf_map[model] = buf
        return buf

    def clear_bufs_by_key(self, buf_keys: list[pathlib.Path | str]):
        """Clear models from buffer cache.

        Parameters
        ----------
        buf_keys : list[pathlib.Path  |  str]
            path to model files. These are used as access keys
        """
        for key in buf_keys:
            self._buf_map.pop(key, None)


class ShapeNetDataset():
    """Helper class for ShapeNet dataset"""
    def __init__(self, base_path):
        # Loaded subdir zips
        self.loaded_synsets = {}

        self.base_path = base_path = pathlib.Path(base_path)
        print(f"Looking for ShapeNet in {base_path}")

        # Search for ShapeNet dir
        unpacked_dirs = list(base_path.glob('ShapeNetCore.v*/'))

        if len(unpacked_dirs) > 0:
            print("Found unpacked ShapeNet core dir")
            self.is_shapenet_zip = False

            # look if subdirs are zips

            zips = list(unpacked_dirs[-1].glob("*.zip"))
            self.is_subdirs_zip = len(zips) != 0
            self.data_path = unpacked_dirs[-1]
            if self.is_subdirs_zip:
                print("ShapeNet synsets are zip files")
            else:
                print("ShapeNet synsets are not zip files")
            return

        # Search for ShapeNet zip
        shapenet_zip = list(base_path.rglob('ShapeNetCore.*.zip'))

        assert len(shapenet_zip) != 0, "Neither ShapeNet dir nor ShapeNet zipfile could be found!"
        print("Found ShapeNet zip file")
        self.is_shapenet_zip = True
        # Read ShapeNet zipfile
        print("Indexing ShapeNet zip archive. This may take some time...")
        self.archive = zipfile.ZipFile(shapenet_zip[-1], 'r')
        print("Done indexing archive.")
        self.data_path = pathlib.Path(pathlib.Path(self.archive.filelist[0].filename).parts[0])

    def open_model(self, model_hash : str, synset : str) -> trimesh.Trimesh:
        """Load given model as trimesh mesh.

        Parameters
        ----------
        model_hash : str
            model id in ShapeNet
        synset : str
            model category as synset

        Returns
        -------
        trimesh.Trimesh
            Loaded model without extra processing
        """
        buf = self.get_model_buf(model_hash, synset)
        t = trimesh.load_mesh(
            buf,
            maintain_order=False,
            skip_materials=True,
            process=False,
            file_type="obj"
        )

        if isinstance(t, trimesh.Scene):
            t = trimesh.util.concatenate(list(t.geometry.values()))
        return t
    
    def get_model_buf(self, model_hash : str, synset : str) -> BytesIO:
        """Read model as buffer.

        Parameters
        ----------
        model_hash : str
            model id to resolve the correct model path
        synset : str
            category synset to resolve the correct model path

        Returns
        -------
        BytesIO
            content of the modelfile stored in a memory buffer
        """
        # if is complete zip, load from path in archive
        if self.is_shapenet_zip:
            model_path = self.data_path / synset / model_hash / 'models' / 'model_normalized.obj'
            return BytesIO(self.archive.read(str(model_path)))

        # if subdir is zip, get subdir archive, then load from relative path    
        if self.is_subdirs_zip:
            model_path = f"{synset}/{model_hash}/models/model_normalized.obj"
            # Load subdir archive if not already cached
            archive = self.loaded_synsets.get(synset, None)
            if archive is None:
                archive = zipfile.ZipFile(f"{self.data_path/synset}.zip")
                self.loaded_synsets[synset] = archive
            return BytesIO(archive.read(model_path))
        
        # if is all extracted, read from path
        model_path = self.data_path / synset / model_hash / 'models' / 'model_normalized.obj'
        with open(str(model_path), 'rb') as fs:
            buf = BytesIO(fs.read())
        return buf


class ModelHandler:
    def __init__(self, anno_id: str, partnet_dataset: PartNetDataset, shapenet_dataset: ShapeNetDataset):
        self.anno_id = anno_id
        self.partnet_dataset = partnet_dataset
        self.shapenet_dataset = shapenet_dataset

        segments_meta, self.meta = partnet_dataset.get_meta(anno_id)
        self.cat_id = CAT2SYNSET[self.meta['model_cat'].lower()]
        self.model_paths = self.partnet_dataset.get_segment_paths(anno_id)

        self.seg_label_map = {}
        self.resolve_seg_labels(segments_meta)

    def resolve_seg_labels(self, segments : dict):
        """scan all children in the segment dict and save assigned labels.

        Parameters
        ----------
        segments : dict
            segment meta information to parse for objs
        """
        # Search through json and save labels to model paths
        for segment in segments:
            label = LABLEMAP[segment['name']]
            # start recursive search, because some segment children have children, too!
            self.recursive_search(label, segment)
        
    def recursive_search(self, label : int, to_parse : dict) -> None:
        """recursively resolves all children containing obj files and assigns them labels

        Parameters
        ----------
        label : int
            parent segment label
        to_parse : dict
            child to parse recursively
        """
        if 'children' in to_parse.keys():
            for child in to_parse['children']:
                self.recursive_search(label, child)
        else:
            for model in to_parse['objs']:
                for model_path in self.model_paths:
                    if pathlib.Path(model_path).stem == model:
                        self.seg_label_map[model_path] = label

    @property
    def partnet_segments(self) -> Generator[tuple[trimesh.Trimesh, int], None, None]:
        for model, label in self.seg_label_map.items():
            yield self.partnet_dataset.open_model(model), label

    @property
    def partnet_bytes_segments(self):
        for model, label in self.seg_label_map.items():
            yield self.partnet_dataset.get_model_buf(model), label
    
    @property
    def shapenet_mesh(self):
        return self.shapenet_dataset.open_model(self.meta['model_id'], self.cat_id)
    
    def close(self):
        self.partnet_dataset.clear_bufs_by_key(self.model_paths)

def rot_mat(angle : float) -> np.ndarray:
    """Constructs homogeneous rotation matrix around y-axis with `angle`.

    Parameters
    ----------
    angle : float
        angle of rotation

    Returns
    -------
    np.ndarray
        4x4 homogeneous transformation matrix
    """    

    return np.array([
        np.cos(angle), 0, np.sin(angle), 0,
        0, 1, 0, 0,
        -np.sin(angle), 0, np.cos(angle), 0,
        0, 0, 0, 1
    ]).reshape(4, 4)


