import datasets
from PIL import Image
import os
from datasets.tasks import ImageClassification

_CITATION = """
描述数据集来源、作者、权利范围
"""

_DESCRIPTION = """
描述数据集的特征
"""


_URLs = {
    "train_set": "load_archive_data/train_set.zip",
    "test_set": "load_archive_data/test_set.zip",
}

_URLss = 'load_archive_data/train_set.zip'



_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

class FullVersion(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="dataset-name1",
            version=datasets.Version("1.0.0"),
            description=_DESCRIPTION,
        ),
        datasets.BuilderConfig(
            name="dataset-name2",
            version=datasets.Version("2.0.0"),
            description=_DESCRIPTION,
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    # 等效 datasets.features.ClassLabel(10)
                    "label": datasets.features.ClassLabel(names=_NAMES),
                }
            ),
            citation=_CITATION,
            task_templates=[ImageClassification(
                image_column="image",
                label_column="label",
            )],
        )

    def _split_generators(self, dl_manager):
        # 返回数据集的路径
        # download_and_extract_path = dl_manager.iter_files(_URLss)
        # download_and_extract_path = dl_manager.iter_archive(_URLss)
        # download_and_extract_path = dl_manager.extract(_URLs)
        # download_and_extract_path = dl_manager.download(_URLss)
        download_and_extract_path = dl_manager.download_and_extract(_URLs)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": download_and_extract_path['train_set'],
                    "split": "train_set",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": download_and_extract_path['test_set'],
                    "split": "test_set",
                },
            ),
        ]




    def _generate_examples(self, filepath, split):
        count = 0
        for tmp_folder in os.listdir(os.path.join(filepath, split)):
            for path in os.listdir(os.path.join(filepath, split, tmp_folder)):
                datafile = os.path.join(filepath, split, tmp_folder, path)
                count += 1
                if split == 'test_set':
                    yield count, {
                        "image": Image.open(datafile),
                        "label": os.path.basename(tmp_folder).lower(),
                    }
                else:
                    yield count, {
                        "image": Image.open(datafile),
                        "label": os.path.basename(tmp_folder).lower(),
                    }
