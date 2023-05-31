# coding=utf-8

import datasets
from PIL import Image
import os

_URL = "load_archive_data/train_set.zip"


class LiteVersion(datasets.GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo()

    def _split_generators(self, dl_manager):
        # 返回数据集的路径
        download_and_extract_path = dl_manager.download_and_extract(_URL)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": download_and_extract_path,
                },
            ),
        ]

    def _generate_examples(self, filepath):
        split = 'train_set'
        count = 0
        for tmp_folder in os.listdir(os.path.join(filepath, split)):
            for path in os.listdir(os.path.join(filepath, split, tmp_folder)):
                datafile = os.path.join(filepath, split, tmp_folder, path)
                count += 1
                yield count, {
                    "image": Image.open(datafile),
                    "label": tmp_folder,
                    }
