### pCR Classification

[toc]

#### Authors

Xuanhe Er, Zhunan Li, Hao Li,Kun Zhu, Yuan Dai
Origanisation: University of Nottingham 🏫

#### Quick Start

##### Install

This repository is tested on Python 3, PyTorch 1.21.1, sklearn 1.1.3 and pandas 1.5.1.

##### Testing

1. put new test data excel file into folder **./data/**
2. cover the original file with this file renamed to 'testDatasetExample.xls'
3. run './FinalTestPCR.py' file
   > python FinalTestPCR.py
4. the prediction output excel file will be the file with path './FinalResult.xls'

##### Training

There are 2 types of training in this project:

1. normal training
   > python train.py
2. K-fold cross validation
   > python train_cross_val.py
