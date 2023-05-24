# -*- coding:utf-8 -*-

import json
from datasets import load_dataset
import torch
import numpy as np
from transformers import TrainingArguments,Trainer,AutoTokenizer,BertTokenizer
tokenizer = BertTokenizer.from_pretrained('input_model/multi_bert',do_lower_case=False)
from modeling import Bert_joint_CRF
from datasets import load_metric
from run_classifier_dataset_utils import write_result

metric = load_metric("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    domain,intent1,intent2,slots
    predictions = np.argmax(logits, axis=-1)
    domain_metrics = metric.compute()
    print(f'The domain matrics is {domain_metrics}')
    return metric.compute(predictions=predictions, references=labels)

from run_classifier_dataset_utils import processors, convert_examples_to_features, write_resultt,write_result
ori = ['Name', 'Number', 'Endloc_poi', 'Startloc_poi', 'Startdate_time', 'Map_mode', 'Position', 'Action', 'Cuisine',
       'Score', 'Schedule', 'Place', 'Preference', 'Date_of_issue', 'Diet', 'Theme', 'Site', 'Time_period',
       'Source_language', 'City', 'Language_category', 'Weather', 'Genre', 'Singer', 'Wind_speed', 'Depature',
       'Temperature', 'Target_language', 'Album', 'Extent', 'Category', 'Date', 'Feeling', 'Mode', 'Stock', 'APP',
       'Culture', 'Time', 'Traffic', 'Star', 'App', 'Price', 'Distance_mode', 'Region', 'Operate', 'Song',
       'Destination', 'Entertainment', 'Obj', 'Facility', 'Encyclopedias', 'Seat_position', 'Object', 'Time_point',
       'Window_position']
intent1 = ['Continue', 'Enlarge', 'Sort', 'Close', 'Vague', 'Save', 'Navigation', 'Narrow', 'Adjustment', 'Delete',
           'Query', 'Open']
# intent2=['Dial','Trunk', 'Hotel', 'Side_window', 'Lunar_calendar', 'Cruise', 'Meteorology', 'Poi', 'People', 'Constellation', 'Playing_mode', 'Weather', 'Sunrise_sunset', 'Mid', 'Wiper', 'Solar_calendar', 'E_dog', 'Oil_quantity', 'Seat_heating', 'Chair', 'Wifi', 'Lighting', 'Panorama', 'Company', 'Music', 'Restaurant', 'Reaview_mirror', 'Translation', 'Holiday', 'Week', 'Time', 'Tyre_pressure', 'Map', 'Alarm_clock', 'Location', 'Route', 'Door', 'Blue_tooth', 'Map_mode', 'Schedule', 'Preference', 'Air_quality', 'Collect', 'Driving_mode', 'Stock', 'Air_temperature', 'Parking', 'Endurance', 'Traffic', 'App', 'Orientation', 'Encyclopedias', 'Command', 'Wind_force', 'Skylight', 'Remaining_time', 'Back', 'Wind_direction', 'Radio', 'Air_speed', 'Massage', 'Seat_ventilation', 'Calculation', 'Air_conditioner', 'Brightness', 'UV_index', 'Volume', 'Serial_No', 'Near', 'Temperature', 'Whole_course', 'Navigation', 'Home', 'Chinese_animal', 'Other', 'Destination', 'Remaining_distance', 'Travel_restriction'
#          ,'Flight','Train','Vague']
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

processor = processors['nlu']()
# train_examples = processor.get_train_examples('train_4_27_1.json')
# train_features = convert_examples_to_features(train_examples, domain_map, intent1_map,intent2_map, slots_map, 128, tokenizer)

def preprocess_function(train_features):
    inputs = {}
    if 'slots' in train_features[0]:
        for i in train_features:
            temp_slots = {}
            for j,k in i['slots'].items():
                if k:
                    temp_slots[j]=k
            i['slots'] = temp_slots
        train_examples = processor.get_train_examples_for_hugging(train_features)
        train_features = convert_examples_to_features(train_examples, domain_map, intent1_map, intent2_map, slots_map, 128,tokenizer)
        inputs['text_ids'] = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        inputs['mask_ids'] = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        inputs['segment_ids'] = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        inputs['domain_ids'] = torch.tensor([f.domain_id for f in train_features], dtype=torch.long)
        inputs['intent1_ids'] = torch.tensor([f.intent1_id for f in train_features], dtype=torch.long)
        inputs['intent2_ids'] = torch.tensor([f.intent2_id for f in train_features], dtype=torch.long)
        inputs['slots_ids'] = torch.tensor([f.slots_id for f in train_features], dtype=torch.long)
    else:
        train_examples = processor.get_test_examples_for_hugging(train_features)
        train_features = convert_examples_to_features(train_examples, domain_map, intent1_map, intent2_map, slots_map, 128,tokenizer)
        inputs['text_ids'] = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        inputs['mask_ids'] = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        inputs['segment_ids'] = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)

    return inputs
# import json
# temp = json.load(open('train_4_27_1.json',encoding='utf-8'))
# temp = temp[:100]
# oo = open('4_27_2.json','w',encoding = 'utf-8')
# json.dump(temp,oo,ensure_ascii=False)
raw_datasets = load_dataset('json', data_files={'train': '4_27_2.json','test':'test_new.json'})
# test_datasets = load_dataset('json', data_files={'test':'test_new.json'})

# data_set = raw_datasets['train'].train_test_split(test_size=0.3)
# train_datasets = raw_datasets['train'].map(preprocess_function)

training_args = TrainingArguments(
    # use_legacy_prediction_loop=True,
    remove_unused_columns=False,
    output_dir = 'trainer_output',
    per_device_train_batch_size=8,
    # per_device_eval_batch_size=8,
    num_train_epochs=1,
    report_to=['tensorboard'],
    learning_rate=2e-05,
    weight_decay=0.001,
    do_train=True,)
    # evaluation_strategy="epoch",)

label_list = processor.get_labels()
model = Bert_joint_CRF.from_pretrained('input_model/multi_bert/pytorch_model.bin', 'input_model/multi_bert/config.json', label_list=label_list,
                                       max_seq_len=128)


trainer = Trainer(
    model=model,
    data_collator = preprocess_function,
    args=training_args,                  # training arguments, defined above
    train_dataset=raw_datasets['train'],         # training dataset
    # eval_dataset=raw_datasets['train'],            # evaluation dataset
    # compute_metrics=compute_metrics

)

# from run_classifier_dataset_utils import InputExample
# ceshi = '打开车门'
# tt = [InputExample(text_a = ceshi,guid = 0)]
#
# pred_features = convert_examples_to_features(
#    tt, domain_map, intent1_map, intent2_map, slots_map, 128, tokenizer)
# all_input_ids = torch.tensor([f.input_ids for f in pred_features], dtype=torch.long)
# all_input_mask = torch.tensor([f.input_mask for f in pred_features], dtype=torch.long)
# all_segment_ids = torch.tensor([f.segment_ids for f in pred_features], dtype=torch.long)
#
trainer.train()
trainer.evaluate(raw_datasets['train'])
result = trainer.predict(raw_datasets['test'])
domain_logits,intent1_logits,intent2_logits,slots_logits = result.predictions[0],result.predictions[1],result.predictions[2],result.predictions[3]

text_list = []
preds = []
raw_test = json.load(open('test_new.json',encoding='utf-8'))
for i in range(len(domain_logits)):
    preds.append({"domain": domain_logits[i], "intent1": intent1_logits[i], 'intent2': intent2_logits[i], "slots": slots_logits[i]})
    text_list.append(raw_test[i]['text'])
write_result(preds, text_list, domain_map, intent1_map, intent2_map,slots_map)





