import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import TensorDataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import Data_Preprocessing
import time
start = time.time()


################## 총 9명의 subject
################## 1명의 subject에 각채널(22채널)당 모델을 만들어서(22개의 모델) 학습을 진행
class DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(625, 100)
        self.linear2 = nn.Linear(100, 20)
        self.linear3 = nn.Linear(20, 4)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.linear3(x)



accu = list()
accu_sum = 0
for z in range(1,10):
    z = str(z)
    train_file = np.load('.\\Preprocessed_Data\\A0'+z+'T_ChannelFirst.npz')
    test_file = np.load('.\\Preprocessed_Data\\A0'+z+'E_ChannelFirst.npz')
    x_train = torch.Tensor(train_file['x'])
    y_train = torch.LongTensor(train_file['y'])
    x_test = torch.Tensor(test_file['x'])
    y_test = torch.LongTensor(test_file['y'])


    model = list()
    for i in range(1, 23):
        i = str(i)
        model.append('model' + i)

    for i in range(0, 22):
        model[i] = DNN()

    for x in range(0, 22):
        xx_train = x_train[x]  ##chaannel : x
        ds_train = TensorDataset(xx_train, y_train)
        loader_train = DataLoader(ds_train, batch_size=48, shuffle=True)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model[x].parameters(), lr=0.001)

        for epoch in range(500):
            for a, b in loader_train:
                pred = model[x](a)
                loss = loss_fn(pred, b - 1)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        print(f'Channel : {x}  Loss : {loss}')

    torch.save(model, '.\\Save_Model\\DNN_subject' + z + '.pth')
    # model = torch.load('.\\Save_Model\\DNN_subject'+z+'.pth')

    with torch.no_grad():
        for x in range(0, 22):
            cnt_tensor = torch.zeros((288, 4))  #### 채널 통합 Tensor

            xx_test = x_test[x]
            ds_test = TensorDataset(xx_test, y_test)
            loader_test = DataLoader(ds_test, batch_size=48, shuffle=False)

            turn = 0
            for a, b in loader_test:
                pred = model[x](a)
                pred = torch.argmax(pred, dim=1)
                for k in range(0, 48):
                    num = pred[k]
                    cnt_tensor[k + turn * 48][num] += 1
                turn += 1

    pred = torch.argmax(cnt_tensor, dim=1)
    correct = pred.eq((y_test - 1).data.view_as(pred)).sum()
    accuracy = correct / 288
    accu_sum += accuracy
    accu.append(accuracy)
    print(f'Accuracy of {z} Data : {accuracy}')



sub = list()
for i in range(1,10):
    i = str(i)
    sub.append('subject'+i)
accu.append(accu_sum / 9)
sub.append('average')

plt.bar(sub, accu)
plt.xlabel('Subject')
plt.ylabel('Accuracy')
plt.show()




















######################################## 모델 기초
# train_file = np.load('.\\Preprocessed_Data\\A08T_ChannelFirst.npz')
# test_file = np.load('.\\Preprocessed_Data\\A08E_ChannelFirst.npz')
# x_train = torch.Tensor(train_file['x'])
# y_train = torch.LongTensor(train_file['y'])
# x_test = torch.Tensor(test_file['x'])
# y_test = torch.LongTensor(test_file['y'])
#
#
# class DNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.linear1 = nn.Linear(625, 100)
#         self.linear2 = nn.Linear(100, 20)
#         self.linear3 = nn.Linear(20, 4)
#
#     def forward(self, x):
#         x = F.relu(self.linear1(x))
#         x = F.relu(self.linear2(x))
#         return self.linear3(x)
#
#
# model = list()
# for i in range(1, 23):
#     i = str(i)
#     model.append('model' + i)
#
# for i in range(0, 22):
#     model[i] = DNN()
#
# for x in range(0, 22):
#     xx_train = x_train[x]  ##channel : x
#     ds_train = TensorDataset(xx_train, y_train)
#     loader_train = DataLoader(ds_train, batch_size=48, shuffle=True)
#
#     loss_fn = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model[x].parameters(), lr=0.01)
#
#     for epoch in range(20):
#         for a, b in loader_train:
#             pred = model[x](a)
#             loss = loss_fn(pred, b - 1)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
######################################## 모델 기초



######################################## 채널별 Accuracy
# with torch.no_grad():
#     for x in range(0,1):
#         xx_test = x_test[x]
#         ds_test = TensorDataset(xx_test, y_test)
#         loader_test = DataLoader(ds_test, batch_size=48, shuffle=False)
#
#         correct = 0
#         for a, b in loader_test:
#             pred = model[x](a)
#             pred = torch.argmax(pred,dim=1)
#             correct += pred.eq(b-1).sum()
#
#         accuracy = correct/288
#         print(f'Channel {x+1} accuracy : {accuracy}')
######################################## 채널별 Accuracy


######################################## 채널 통합 Accuracy
# with torch.no_grad():
#     for x in range(0, 22):
#         cnt_tensor = torch.zeros((288, 4))  #### 채널 통합 Tensor
#
#         xx_test = x_test[x]
#         ds_test = TensorDataset(xx_test, y_test)
#         loader_test = DataLoader(ds_test, batch_size=48, shuffle=False)
#
#         turn = 0
#         for a, b in loader_test:
#             pred = model[x](a)
#             pred = torch.argmax(pred, dim=1)
#             for k in range(0, 48):
#                 num = pred[k]
#                 cnt_tensor[k + turn * 48][num] += 1
#             turn += 1
#
# pred = torch.argmax(cnt_tensor, dim=1)
# correct = pred.eq((y_test - 1).data.view_as(pred)).sum()
# accuracy = correct / 288
# print(f'Accuracy of All : {accuracy}')
######################################## 채널 통합 Accuracy




######################################## 모델 저장 및 불러오기
# torch.save(model,'.\\Save_Model\\model.pth')
# model = list()
# for i in range(1, 23):
#     i = str(i)
#     model.append('model' + i)
#
# for i in range(0, 22):
#     model[i] = DNN()
#
# model = torch.load('.\\Save_Model\\model.pth')
#
# with torch.no_grad():
#     for x in range(0, 22):
#         cnt_tensor = torch.zeros((288, 4))  #### 채널 통합 Tensor
#
#         xx_test = x_test[x]
#         ds_test = TensorDataset(xx_test, y_test)
#         loader_test = DataLoader(ds_test, batch_size=48, shuffle=False)
#
#         turn = 0
#         for a, b in loader_test:
#             pred = model[x](a)
#             pred = torch.argmax(pred, dim=1)
#             for k in range(0, 48):
#                 num = pred[k]
#                 cnt_tensor[k + turn * 48][num] += 1
#             turn += 1
#
# pred = torch.argmax(cnt_tensor, dim=1)
# correct = pred.eq((y_test - 1).data.view_as(pred)).sum()
# accuracy = correct / 288
# print(f'Accuracy of All : {accuracy}')
######################################## 모델 저장 및 불러오기


######################################## 시각화
# x = ['subject8']
# y = accuracy
# plt.bar(x,y)
# plt.xlabel('Subject')
# plt.ylabel('Accuracy')
# plt.show()
######################################## 시각화




t = time.time() - start
if t > 60:
    print(f'{t / 60:.2f} Min')
else:
    print(f'{t:.2f} Sec')
