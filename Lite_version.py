<<<<<<< HEAD
# coding=utf-8

import datasets
from PIL import Image
import os

# 单文件
=======
import datasets
from PIL import Image
import os
import numpy as np
from datasets.info import SupervisedKeysData, PostProcessedInfo
#
# datafile = r"C:\Users\JC\.cache\huggingface\datasets\downloads\extracted\72cefad2817f4411a2774eabf96cf0c9eb47b33b998b57977251d30c3f0658b4\train_set\0\29.png"
# print(np.array(Image.open(datafile).convert("RGB")).astype(int).shape)

# 单文件
# 多文件
>>>>>>> 793a0c4 (测试)
_URL = "load_archive_data/train_set.zip"


class LiteVersion(datasets.GeneratorBasedBuilder):
    """
<<<<<<< HEAD
    'LiteVersion':类名称可以随意替换，即你对数据集的命名
    """

    def _info(self):
        return datasets.DatasetInfo()
=======
    'LiteVersion':类名称可以随意替换
    """

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    # "image": datasets.Value("int32"),
                    "image": datasets.Image(),
                    "label": datasets.features.ClassLabel(names=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]),
                },
            ),
            supervised_keys=('1',),
        )
>>>>>>> 793a0c4 (测试)

    def _split_generators(self, dl_manager):
        # 返回数据集的路径
        download_and_extract_path = dl_manager.download_and_extract(_URL)
<<<<<<< HEAD
=======
        # download_and_extract_path = dl_manager.download_and_extract(_URLS)
>>>>>>> 793a0c4 (测试)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": download_and_extract_path,
                },
            ),
        ]

    def _generate_examples(self, filepath):
<<<<<<< HEAD
        # _generate_examples 中输入的参数就是 _split_generators 中 gen_kwargs 的参数
=======
>>>>>>> 793a0c4 (测试)
        split = 'train_set'
        count = 0
        for tmp_folder in os.listdir(os.path.join(filepath, split)):
            for path in os.listdir(os.path.join(filepath, split, tmp_folder)):
                datafile = os.path.join(filepath, split, tmp_folder, path)
                count += 1
                yield count, {
<<<<<<< HEAD
=======
                    # "image": np.array(Image.open(datafile).convert("RGB")),
>>>>>>> 793a0c4 (测试)
                    "image": Image.open(datafile),
                    "label": tmp_folder,
                }
