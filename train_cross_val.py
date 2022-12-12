# -*- coding: utf-8 -*-
# @Author : Xuanhe Er
# @Time   : 03/12/2022 16:20

import numpy as np
import torch
import torch.utils.data as Data
import time
from sklearn.model_selection import KFold

import patients_classification_data_loader
import model

k_folds = 5
# result dict of accuracy, used to calculate the MSE
results = {}
epoches = 250
lr = 1

# load data
patient_dataset = patients_classification_data_loader.PatientsDataLoader('./data/trainDataset.xls')

kfold = KFold(n_splits=k_folds, shuffle=True)

for fold, (train_ids, test_ids) in enumerate(kfold.split(patient_dataset)):

    print(f'FOLD {fold}')
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

    # split test dataset and train dataset and load the data
    #     torch_train_dataset = Data.TensorDataset(data_tensor[28:], label_tensor[28:])
    loader = Data.DataLoader(
        dataset=patient_dataset,
        batch_size=15,
        sampler=train_subsampler
    )

    # torch_test_dataset = Data.TensorDataset(data_tensor[0:28], label_tensor[0:28])
    loader2 = Data.DataLoader(
        dataset=patient_dataset,
        batch_size=15,
        sampler=test_subsampler
    )
    mdl = model.NN()

    # train the data
    start_time = time.time()
    print('start training...')
    print('learning rate =', lr)
    loss_func = torch.nn.CrossEntropyLoss()
    optimzer = torch.optim.SGD(params=mdl.parameters(), lr=lr)

    for epoch in range(epoches):
        current_loss = 0.0
        # print(f' The epoch is {epoch + 1}')
        for step, (batch_x, batch_label) in enumerate(loader):
            optimzer.zero_grad()
            output = mdl(batch_x)
            loss = loss_func(output, batch_label)
            loss.backward()
            optimzer.step()
            current_loss += loss.item()
            if step % 50 == 49:
                print('Loss after SGD %5d: %.3f' %
                      (step + 1, current_loss))
                current_loss = 0.0
    end_time = time.time()
    print('total cost of time:', end_time - start_time, 'secs')

    # Process is complete.
    print('training finished. Saving model.')
    # Print about testing
    print('TESTING')

    # Saving the model
    save_path = f'./savedmodels/model-fold-{fold}.pth'
    torch.save(mdl.state_dict(), save_path)

    # test the data
    correct = 0
    total = 0
    with torch.no_grad():
        for step, (batch_dataMat, batch_labelMat) in enumerate(loader2, 0):
            output = mdl(batch_dataMat)
            _, p = torch.max(output.data, 1)
            total += batch_labelMat.size(0)
            correct += (p == batch_labelMat).sum().item()
            # Print accuracy
        print('Accuracy for fold %d: %d %%' % (fold, 100.0 * correct / total))
        print('--------------------------------')
        results[fold] = 100.0 * (correct / total)

# Print fold results
print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
print('--------------------------------')
sum = 0.0
for key, value in results.items():
    print(f'Fold {key}: {value} %')
    sum += value
print(f'Average: {sum / len(results.items())} %')
