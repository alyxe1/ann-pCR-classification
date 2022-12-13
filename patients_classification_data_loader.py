# -*- coding: utf-8 -*-
# @Author : Xuanhe Er
# @Time   : 30/11/2022 21:05

from torch.utils.data.dataset import Dataset

import preprocessing_classification
import torch

torch.set_default_tensor_type(torch.DoubleTensor)


class PatientsDataLoader(Dataset):
    def __init__(self, file_path):
        x, y = preprocessing_classification.data_preprocessing(file_path)

        self.data = x
        self.label = y

    def __getitem__(self, item):
        return self.data[item], self.label[item]

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    x, y = preprocessing_classification.data_preprocessing('./data/trainDataset.xls')
    print(x)
    print(y)
