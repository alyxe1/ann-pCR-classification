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

SAVED_MODEL_PATH = "./savedmodels/finalmodel.pth"
TEST_EXCEL_PATH = "./data/testDatasetExample.xls" #

LDA_MODEL_PATH = "./savedmodels/lda_model.pkl"
SCALER_MODEL_PATH = "./savedmodels/scaler.pkl"
NORM_MODEL_PATH = "./savedmodels/normalizer.pkl"

SAVED_EXCEL_PATH = "./FinalResult.xls"


def test_data_preprocessing(test_path):
    df_test = preprocessing_classification.load_data(test_path)
    df_test = preprocessing_classification.data_cleansing(df_test)

    x_test = df_test.to_numpy()
    excel = pd.read_excel(TEST_EXCEL_PATH, skiprows=0, sheet_name='Sheet1')
    x_ID = excel['ID']
    x_test = x_test[:, 1:]  # fake change here to 3 (or 1)

    scaler = load_scaler(SCALER_MODEL_PATH)
    x_test = scaler.transform(x_test)
    return x_test, x_ID


def load_scaler(scaler_path):
    with open(scaler_path, "rb+") as f:
        scaler = pickle.load(f)
        return scaler


def load_lda_model(lda_model_path):
    with open(lda_model_path, "rb+") as f:
        lda = pickle.load(f)
        return lda


def load_norm_model(norm_model_path):
    with open(norm_model_path, "rb+") as f:
        norm_model = pickle.load(f)
        return norm_model


class PatientsTestDataLoader(Dataset):
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
model = NN()
model.load_state_dict(torch.load(SAVED_MODEL_PATH))
model.eval()  # starts the evaluation mode

# load the data
# test_dataset = patients_classification_data_loader.PatientsDataLoader(TEST_EXCEL_PATH)
# test_loader = Data.DataLoader(
#     dataset=test_dataset,
#     batch_size=15
# )
import xlwt


# test the data
def test():
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

# draw xlsx
n = 0
df_res = pd.DataFrame(columns=("ID", "pCR(outcome)"))
for acc in test_result:
    df_res.loc[n] = [torch_test_dataset.x_ID[n], acc]
    n+=1
df_res.to_excel(SAVED_EXCEL_PATH,index=False)

