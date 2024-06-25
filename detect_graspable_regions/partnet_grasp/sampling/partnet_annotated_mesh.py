from detect_graspable_regions.partnet_grasp.sampling.constants import CAT2SYNSET

import json
import os


class PartNetAnnotatedMesh:
    def __init__(self, anno_id: str, base_partnet_path: str, base_shapenet_path: str):
        self.anno_id = anno_id
        self.base_PartNet_path = base_partnet_path
        self.base_ShapeNet_path = base_shapenet_path

        with open(f"{base_partnet_path}/{anno_id}/result.json") as f:
            self.result = json.load(f)[0]
            self.segments = self.result['children']

        with open(f"{base_partnet_path}/{anno_id}/meta.json") as f:
            self.meta = json.load(f)

        self.objs = os.listdir(f"{base_partnet_path}/{anno_id}/objs")

        cat_id = CAT2SYNSET[self.meta['model_cat'].lower()]
        self.ShapeNet_model_path = (
            f"{self.base_ShapeNet_path}/{cat_id}/{self.meta['model_id']}/training/model_normalized.obj"
        )

    def print_paths(self):
        print(self.ShapeNet_model_path)
        print(f"{self.base_PartNet_path}/{self.anno_id}/objs/{self.objs[0]}")
