import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import PIL
import os

class TrainDataset(Dataset):
    def __init__(self, root_path, transform, data_aug=None):
        super(TrainDataset,self).__init__()
        self.augmentation = data_aug
        self.root_path = root_path
        self.transform = transform
        self.image_list = list()
        
        # 파일 경로가 잘 설정되어 있다면 그대로 유지
        self.annotation = pd.read_csv(os.path.join(self.root_path, '../Train_64.csv'))

        self.image_list = np.array(self.annotation.values.tolist())[:, 0]
        self.labels = np.array(self.annotation.values.tolist())[:, 1]

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self,index):
        # PIL.Image.open 사용 시 PIL 대신 PIL.Image를 import하는 것이 더 명확함
        img = PIL.Image.open(os.path.join(self.root_path,str('Train_64'),self.image_list[index]))
        if self.augmentation is not None:
            img = self.augmentation(img)
        img = self.transform(img)
        return img, int(self.labels[index])

class TestDataset(Dataset):
    def __init__(self, root_path, transform, data_aug=None):
        super(TestDataset,self).__init__()
        self.root_path = root_path
        self.transform = transform
        self.image_list = list()
        
        # 파일 경로가 잘 설정되어 있다면 그대로 유지
        self.annotation = pd.read_csv(os.path.join(self.root_path, '../Test_64.csv'))

        self.image_list = np.array(self.annotation.values.tolist())[:, 0]


    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self,index):
        # PIL.Image.open 사용 시 PIL 대신 PIL.Image를 import하는 것이 더 명확함
        img = PIL.Image.open(os.path.join(self.root_path,str('Test_64'),self.image_list[index]))
        img = self.transform(img)
        return img
