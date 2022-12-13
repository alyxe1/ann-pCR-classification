# -*- coding: utf-8 -*-
# @Author : Xuanhe Er
# @Time   : 10/12/2022 23:55
import torch
import torch.utils.data as Data
import pickle
import numpy as np
from torch.utils.data.dataset import Dataset

import torch
from model import NN
import preprocessing_classification
import pandas as pd

# Management of file paths
SAVED_MODEL_PATH = "./savedmodels/finalmodel.pth"
TEST_EXCEL_PATH = "./data/testDatasetExample.xls"  #

LDA_MODEL_PATH = "./savedmodels/lda_model.pkl"
SCALER_MODEL_PATH = "./savedmodels/scaler.pkl"
NORM_MODEL_PATH = "./savedmodels/normalizer.pkl"

SAVED_EXCEL_PATH = "./FinalResult.xls"


def test_data_preprocessing(test_path):
    """
    preprocessing for test dataset
    :param test_path: path of test excel file
    :return: x after preprocessing, id of x
    """
    df_test = preprocessing_classification.load_data(test_path)
    df_test = preprocessing_classification.data_cleansing(df_test)

    x_test = df_test.to_numpy()
    excel = pd.read_excel(TEST_EXCEL_PATH, skiprows=0, sheet_name='Sheet1')
    x_ID = excel['ID']
    x_test = x_test[:, 1:]
    # load the scaler of Max-min Scaler
    scaler = load_scaler(SCALER_MODEL_PATH)
    # transforms them
    x_test = scaler.transform(x_test)
    return x_test, x_ID


def load_scaler(scaler_path):
    """
    deserialisation saved scaler model
    :param scaler_path: file path of deserialisation saved scaler model
    :return: scaler
    """
    with open(scaler_path, "rb+") as f:
        scaler = pickle.load(f)
        return scaler


def load_lda_model(lda_model_path):
    """
    deserialisation saved LDA model
    :param scaler_path: file path of deserialisation saved LDA model
    :return: lda model
    """
    with open(lda_model_path, "rb+") as f:
        lda = pickle.load(f)
        return lda


def load_norm_model(norm_model_path):
    """
    deserialisation saved normaliser
    :param scaler_path: file path of deserialisation saved normaliser
    :return: normaliser
    """
    with open(norm_model_path, "rb+") as f:
        norm_model = pickle.load(f)
        return norm_model


class PatientsTestDataLoader(Dataset):
    """
    customised dataset defined by programmer
    """

    def __init__(self, test_path):
        scaled_x, x_ID = test_data_preprocessing(test_path)
        norn_model = load_norm_model(NORM_MODEL_PATH)
        scaled_normed_x = norn_model.transform(scaled_x)
        lda = load_lda_model(LDA_MODEL_PATH)
        self.data = lda.transform(scaled_normed_x)
        self.x_ID = x_ID

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


torch_test_dataset = PatientsTestDataLoader(TEST_EXCEL_PATH)
test_loader = Data.DataLoader(
    dataset=torch_test_dataset,
    batch_size=15
)
model = NN()  # create model
model.load_state_dict(torch.load(SAVED_MODEL_PATH))  # load neural network model
model.eval()  # starts the evaluation mode(compulsory)


# test the data
def test():
    """
    to estimate the data and save them into ./FinalResult.xls
    :return: list of predicted value
    """
    result = []
    with torch.no_grad():
        for step, batch_dataMat in enumerate(test_loader, 0):
            output = model(batch_dataMat.float())
            _, p = torch.max(output.data, 1)
            if p is not None:
                for item in p:
                    result.append(item.item())
                print(f"prediction value is {p}")
    return result


test_result = test()

# save examples of Id and predicted y into xlsx
n = 0
df_res = pd.DataFrame(columns=("ID", "pCR(outcome)"))
for acc in test_result:
    df_res.loc[n] = [torch_test_dataset.x_ID[n], acc]
    n += 1
# save
df_res.to_excel(SAVED_EXCEL_PATH, index=False)
