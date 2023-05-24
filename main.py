from datasets import load_dataset

'''
Lite Version:
    仅保留生成数据
    
Full Version:
    包含较为完整的info,config等信息
    
根据实际需求自行修改path
'''


# data = load_dataset('./Lite_version.py')

data = load_dataset('cifar10/cifar10.py')
print(data)

# data = load_dataset('Full_version.py', 'dataset-name2')

data = load_dataset('Full_version.py', 'dataset-name1')

# data = load_dataset(r'Reference_py_documentation\beans.py')
print(data['train'].features["label"])
# print(type(data[ 'train'][0]['label']))

