# -*- coding: utf-8 -*-
# @Author : Xuanhe Er
# @Time   : 30/11/2022 22:55
import pandas as pd
import numpy as np
import torch
import pickle
from sklearn.preprocessing import MinMaxScaler, normalize
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def data_preprocessing(file_path):
    data = load_data(file_path)
    data = data_cleansing(data)
    data = data.to_numpy()
    x, y = data[:, 3:], data[:, 1]

    x = scale(x) # Data Transformation: Max-min scaler
    y = y.astype('int')

    x = LDA(x, y) # Data Reduction: Linear Discrimination Analysis
    return x, y

def LDA(x, y):
    """
    LDA
    :param x:
    :param y:
    :return:
    """
    x_norm = normalize(x, norm='l2')
    lda = LinearDiscriminantAnalysis(n_components=1)
    x_new = lda.fit_transform(x_norm, y)
    with open("./savedmodels/lda_model.pkl", "wb") as f:
        pickle.dump(lda, f)
    return x_new


def scale(x):
    """
    Max-min Scaler
    :param x:
    :return:
    """
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)
    with open("./savedmodels/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    return x

def load_data(file_path):
    """
    data loader function
    :param file_path:
    :return:
    """
    patients = pd.read_excel(file_path, skiprows=0, sheet_name='Sheet1')
    return patients
def data_cleansing(data):
    """"""
    return drop_none(data)
def drop_none(data):
    try:
        data.loc[data['pCR (outcome)'] == 999] = np.nan
        return data.dropna()
    except:
        print("pCR (outcome) column is not existed!(won't affect the evaluation stage)")
        return data


if __name__ == '__main__':
    # patients_np = load_data('./data/trainDataset.xls')
    # print(x)
    # print(y)
    # data = data_cleansing(patients_np)

    x, y = data_preprocessing('./data/trainDataset.xls')
    print(x)
    print(y)
    print(len(x))
    print(len(y))
    # print(len(x[0]))
    # print(x)
    # print(y, type(y[1]))

