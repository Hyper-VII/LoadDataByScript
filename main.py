from datasets import load_dataset

'''
Lite Version:
    仅保留生成数据
    
Full Version:
    包含较为完整的info,config等信息
'''



data = load_dataset('Lite_version.py')
print(data['train'][:10])

data = load_dataset('Full_version.py', 'dataset-name1')
print(data['train'][:1])