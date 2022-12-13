# -*- coding: utf-8 -*-
# @Author : Xuanhe Er
# @Time   : 03/12/2022 16:20

from torch import nn
import torch.nn.functional as F
import preprocessing_classification
import torch


class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(1, 5)
        self.fc2 = torch.nn.Linear(5, 5)
        self.fc3 = torch.nn.Linear(5, 2)

    def forward(self, x):
        a1 = F.sigmoid(self.fc1(x))
        a2 = F.sigmoid(self.fc2(a1))
        a3 = F.softmax(self.fc3(a2), dim=1)
        return a3


if __name__ == '__main__':
    x, y = preprocessing_classification.data_preprocessing('./data/trainDataset.xls')
    model = NN()
    print(model(x))
