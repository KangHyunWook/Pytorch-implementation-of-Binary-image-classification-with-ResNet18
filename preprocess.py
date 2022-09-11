"""
"""
import os
import shutil
import splitfolders
from random import shuffle

if not os.path.exists('./data'):
    os.rename('train', 'data')

    data_path='./data'

    file_names=os.listdir(data_path)

    file_path_list=[]
    for fn in file_names:
        full_path = os.path.join(data_path, fn)
        file_path_list.append(full_path)
        
    shuffle(file_path_list)

    if not os.path.exists(os.path.join(data_path, 'train')):
        os.mkdir(os.path.join(data_path,'train'))

    if not os.path.exists(os.path.join(data_path, 'val')):
        os.mkdir(os.path.join(data_path,'val'))

    if not os.path.exists(os.path.join(data_path, 'test')):
        os.mkdir(os.path.join(data_path,'test'))    

    train_len=int(0.6*len(file_path_list))
    val_len=int(0.2*len(file_path_list))

    for file_path in file_path_list[:train_len]:
        shutil.move(file_path, os.path.join(data_path,'train'))
        
    for file_path in file_path_list[train_len:train_len+val_len]:
        shutil.move(file_path, os.path.join(data_path,'val'))

    for file_path in file_path_list[train_len+val_len:]:
        shutil.move(file_path, os.path.join(data_path,'test'))    












