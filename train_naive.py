# -*- coding: utf-8 -*-
# @Author : Xuanhe Er
# @Time   : 03/12/2022 16:20

import torch
import torch.utils.data as Data
import time

import model
import preprocessing_classification
epoches = 250
lr = 1
TRAIN_PATH = "./data/trainDataset.xls"
# load data
x, y = preprocessing_classification.data_preprocessing(TRAIN_PATH)
x = torch.from_numpy(x)
y = torch.from_numpy(y)
train_dataset = Data.TensorDataset(x[30:], y[30:])
# split test dataset and train dataset and load the data
#     torch_train_dataset = Data.TensorDataset(data_tensor[28:], label_tensor[28:])
loader = Data.DataLoader(
    dataset=train_dataset,
    batch_size=15
)
test_dataset = Data.TensorDataset(x[0:30], y[0:30])
# torch_test_dataset = Data.TensorDataset(data_tensor[0:28], label_tensor[0:28])
loader2 = Data.DataLoader(
    dataset=test_dataset,
    batch_size=15
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
        batch_x = batch_x.float()
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
save_path = f'./savedmodels/finalmodel.pth'
torch.save(mdl.state_dict(), save_path)

# test the data
correct = 0
total = 0
with torch.no_grad():
    for step, (batch_dataMat, batch_labelMat) in enumerate(loader2, 0):
        batch_dataMat = batch_dataMat.float()
        output = mdl(batch_dataMat)
        _, p = torch.max(output.data, 1)
        total += batch_labelMat.size(0)
        correct += (p == batch_labelMat).sum().item()
        # Print accuracy
    print('Accuracy is %d %%' % (100.0 * correct / total))
    print('--------------------------------')

