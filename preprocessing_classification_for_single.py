# -*- coding: utf-8 -*-
# @Author : Xuanhe Er
# @Time   : 30/11/2022 22:55
import pandas as pd
import numpy as np
import torch
import pickle
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split


def data_preprocessing(file_path):
    """
    load, cleansing, split,
    scaler, LDA, convert to tensors

    :param file_path: data filepath
    :return: x_train_norm, x_test, y_train, y_test
    """
    # 1. load the data
    data = load_data(file_path)
    # 2. data cleansing
    data = data_cleansing(data)
    data = data.to_numpy()
    x, y = data[:, 3:], data[:, 1]

    # 3. split data for training and testing
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.19, random_state=42)
    # 4. scaler: scaler using train_x to fit model
    scaler_model = MinMaxScaler()
    scaler_model.fit(x_train)
    # using scaler model to train x and test x
    x_train = scaler_model.transform(x_train)
    x_test = scaler_model.transform(x_test)
    # save the scaler model
    with open("./savedmodels/scaler.pkl", "wb") as f:
        pickle.dump(scaler_model, f)

    # 5. using Normalize model
    normalizer = Normalizer(norm='l2')
    normalizer.fit(x_train)
    # save the scaler model
    x_train_norm = normalizer.transform(x_train)
    x_test_norm = normalizer.transform(x_test)
    with open("./savedmodels/normalizer.pkl", "wb") as f:
        pickle.dump(normalizer, f)

    # 6. LDA using LDA model to train x and test x
    lda = LinearDiscriminantAnalysis(n_components=1)
    y_train = y_train.astype('int')
    y_test = y_test.astype('int')
    lda.fit(x_train_norm, y_train)
    x_train_norm = lda.transform(x_train_norm)
    x_test_norm = lda.transform(x_test_norm)
    # save the scaler model
    with open("./savedmodels/lda_model.pkl", "wb") as f:
        pickle.dump(lda, f)
    # 7. convert from ndarray to tensors
    x_train_norm = torch.from_numpy(x_train_norm)
    x_test_norm = torch.from_numpy(x_test_norm)
    y_train = torch.from_numpy(y_train)
    y_test = torch.from_numpy(y_test)

    return x_train_norm, x_test_norm, y_train, y_test


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
