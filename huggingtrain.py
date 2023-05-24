import datasets
from PIL import Image
import os
import json
import numpy as np
from datasets.info import SupervisedKeysData, PostProcessedInfo
#
# datafile = r"C:\Users\JC\.cache\huggingface\datasets\downloads\extracted\72cefad2817f4411a2774eabf96cf0c9eb47b33b998b57977251d30c3f0658b4\train_set\0\29.png"
# print(np.array(Image.open(datafile).convert("RGB")).astype(int).shape)

# 单文件
# 多文件
_URLs = {
    "train": "train.json",
    "test": "test.json",
}
ori = ['Name', 'Number', 'Endloc_poi', 'Startloc_poi', 'Startdate_time', 'Map_mode', 'Position', 'Action', 'Cuisine',
       'Score', 'Schedule', 'Place', 'Preference', 'Date_of_issue', 'Diet', 'Theme', 'Site', 'Time_period',
       'Source_language', 'City', 'Language_category', 'Weather', 'Genre', 'Singer', 'Wind_speed', 'Depature',
       'Temperature', 'Target_language', 'Album', 'Extent', 'Category', 'Date', 'Feeling', 'Mode', 'Stock', 'APP',
       'Culture', 'Time', 'Traffic', 'Star', 'App', 'Price', 'Distance_mode', 'Region', 'Operate', 'Song',
       'Destination', 'Entertainment', 'Obj', 'Facility', 'Encyclopedias', 'Seat_position', 'Object', 'Time_point',
       'Window_position']
intent1 = ['Continue', 'Enlarge', 'Sort', 'Close', 'Vague', 'Save', 'Navigation', 'Narrow', 'Adjustment', 'Delete',
           'Query', 'Open']
domain = ['Car_system', 'Knowledge', 'Other', 'Multi_media', 'Travel']
intent2=['Dial','Trunk', 'Hotel', 'Side_window', 'Lunar_calendar', 'Cruise', 'Meteorology', 'Poi', 'People', 'Constellation', 'Playing_mode', 'Weather', 'Sunrise_sunset', 'Mid', 'Wiper', 'Solar_calendar', 'E_dog', 'Oil_quantity', 'Seat_heating', 'Chair', 'Wifi', 'Lighting', 'Panorama', 'Company', 'Music', 'Restaurant', 'Reaview_mirror', 'Translation', 'Holiday', 'Week', 'Time', 'Tyre_pressure', 'Map', 'Alarm_clock', 'Location', 'Route', 'Door', 'Blue_tooth', 'Map_mode', 'Schedule', 'Preference', 'Air_quality', 'Collect', 'Driving_mode', 'Stock', 'Air_temperature', 'Parking', 'Endurance', 'Traffic', 'App', 'Orientation', 'Encyclopedias', 'Command', 'Wind_force', 'Skylight', 'Remaining_time', 'Back', 'Wind_direction', 'Radio', 'Air_speed', 'Massage', 'Seat_ventilation', 'Calculation', 'Air_conditioner', 'Brightness', 'UV_index', 'Volume', 'Serial_No', 'Near', 'Temperature', 'Whole_course', 'Navigation', 'Home', 'Chinese_animal', 'Other', 'Destination', 'Remaining_distance', 'Travel_restriction'
         ,'Flight','Train','Vague']

slots = []
slots.append("O")
for slot in ori:
    slots.append("B-" + slot)
    slots.append("I-" + slot)

domain_map = {}
for (i, label) in enumerate(domain):
    domain_map[label] = i

intent1_map = {}
for (i, label) in enumerate(intent1):
    intent1_map[label] = i

intent2_map = {}
for (i, label) in enumerate(intent2):
    intent2_map[label] = i

slots_map = {}
slots_map_inv={}
slots_resultt = {}
for (i, label) in enumerate(slots):
    slots_map[label] = i
    slots_map_inv[i]=label
    slots_resultt[i]=[0,0]

class Huggingdata(datasets.GeneratorBasedBuilder):
    """
    'LiteVersion':类名称可以随意替换
    """

    def _info(self):
        return datasets.DatasetInfo(
        )

    def _split_generators(self, dl_manager):
        # 返回数据集的路径
        download_and_extract_path = dl_manager.download_and_extract(_URLs)
        # print(download_and_extract_path)
        # download_and_extract_path = dl_manager.download_and_extract(_URLS)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": download_and_extract_path["train"],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": download_and_extract_path["test"],
                    "split": "test"
                },
            ),
        ]

    def _generate_examples(self, filepath, split):

        with open(filepath, encoding='utf-8') as f:
            file = json.load(f)
            for idx, datadict in enumerate(file):
                example = {}
                if split=="test":
                    example["text"] = datadict["text"]
                    example["domain"] = None
                    example["intent1"] = None
                    example["intent2"] = None
                    example["slots"] = None
                    yield idx, example
                else:
                    example["text"] = datadict["text"]
                    example["domain"] = datadict["domain"]
                    example["intent1"] = datadict["intent1"]
                    example["intent2"] = datadict["intent2"]
                    example["slots"] = datadict["slots"]
                    yield idx, example




