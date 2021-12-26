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

import time

start = time.time()


class HS_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel_1 = nn.Sequential(nn.Conv2d(1, 10, kernel_size=(1, 85), stride=(1, 3)),
                                      nn.BatchNorm2d(10),
                                      nn.Conv2d(10, 10, kernel_size=(22, 1), stride=(1, 1)),
                                      nn.BatchNorm2d(10),
                                      nn.ELU(),
                                      nn.MaxPool2d(1,6))  ##10, 1, 31
        self.kernel_2 = nn.Sequential(nn.Conv2d(1, 10, kernel_size=(1, 65), stride=(1, 3)),
                                      nn.BatchNorm2d(10),
                                      nn.Conv2d(10, 10, kernel_size=(22, 1), stride=(1, 1)),
                                      nn.BatchNorm2d(10),
                                      nn.ELU(),
                                      nn.MaxPool2d(1,6))  ##10, 1, 32
        self.kernel_3 = nn.Sequential(nn.Conv2d(1, 10, kernel_size=(1, 45), stride=(1, 3)),
                                      nn.BatchNorm2d(10),
                                      nn.Conv2d(10, 10, kernel_size=(22, 1), stride=(1, 1)),
                                      nn.BatchNorm2d(10),
                                      nn.ELU(),
                                      nn.MaxPool2d(1,6))  ##10, 1, 33
        self.linear1 = nn.Sequential(nn.Linear(2880, 100), nn.ELU(), nn.Dropout(p=0.5))
        self.linear2 = nn.Linear(100, 4)

    def forward(self, x, y, z):
        xyz = None
        for i in {x, y, z}:
            x1 = self.kernel_1(i)
            x1 = x1.view(x1.size(0), -1)
            x2 = self.kernel_2(i)
            x2 = x2.view(x2.size(0), -1)
            x3 = self.kernel_3(i)
            x3 = x3.view(x3.size(0), -1)
            if xyz == None:
                xyz = torch.cat((x1, x2, x3), dim=1)    ##48 ,960
            else:
                xyz = torch.cat((xyz, x1, x2, x3), dim=1)
        x = self.linear1(xyz)  ## 48, 2880 --> 48 200
        x = self.linear2(x)
        return x


accu = list()
accu_sum = 0
learning_rate = 0.01
nb_epoch = 500

for z in range(1,10):
    z = str(z)
    train_file = np.load('.\\Post_Research\\Preprocessed_Data\\3s - 5.5s\\Beta_13-32Hz_BPF\\A0'+z+'T.npz')
    test_file = np.load('.\\Post_Research\\Preprocessed_Data\\3s - 5.5s\\Beta_13-32Hz_BPF\\A0'+z+'E.npz')
    beta_train = torch.FloatTensor(train_file['x'])
    y_train = torch.LongTensor(train_file['y'])
    beta_test = torch.FloatTensor(test_file['x'])
    y_test = torch.LongTensor(test_file['y'])

    train_file = np.load('.\\Post_Research\\Preprocessed_Data\\3s - 5.5s\\Theta_4-7Hz_BPF\\A0'+z+'T.npz')
    test_file = np.load('.\\Post_Research\\Preprocessed_Data\\3s - 5.5s\\Theta_4-7Hz_BPF\\A0'+z+'E.npz')
    theta_train = torch.FloatTensor(train_file['x'])
    theta_test = torch.FloatTensor(test_file['x'])

    train_file = np.load('.\\Post_Research\\Preprocessed_Data\\3s - 5.5s\\Mu_8-13Hz_BPF\\A0'+z+'T.npz')
    test_file = np.load('.\\Post_Research\\Preprocessed_Data\\3s - 5.5s\\Mu_8-13Hz_BPF\\A0'+z+'E.npz')
    mu_train = torch.FloatTensor(train_file['x'])
    mu_test = torch.FloatTensor(test_file['x'])


    beta_train = beta_train.unsqueeze(dim=1)  ##(288, 1, 22, 625) 만들기
    theta_train = theta_train.unsqueeze(dim=1)
    mu_train = mu_train.unsqueeze(dim=1)

    ds_train = TensorDataset(beta_train, theta_train, mu_train, y_train)
    loader_train = DataLoader(ds_train, batch_size=48, shuffle=True)

    model = HS_CNN()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    for epoch in range(nb_epoch):
        error = None
        for x1, x2, x3, y in loader_train:
            model.train()
            pred = model(x1, x2, x3)
            loss = loss_fn(pred, y - 1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            error = loss
        if (epoch+1) % 100 == 0:
            print(f'[Epoch] : {epoch+1:>5d}  [Loss] : {error:>2.10f}')

    with torch.no_grad():
        beta_test = beta_test.unsqueeze(dim=1)
        theta_test = theta_test.unsqueeze(dim=1)
        mu_test = mu_test.unsqueeze(dim=1)
        ds_test = TensorDataset(beta_test, theta_test, mu_test, y_test)
        loader_test = DataLoader(ds_test, batch_size=48, shuffle=False)

        correct = 0
        for x1, x2, x3, y in loader_test:
            model.eval()
            pred = model(x1, x2, x3)
            pred = torch.argmax(pred, dim=1)
            correct += pred.eq(y - 1).sum()

    accuracy = correct / 288
    accu_sum += accuracy
    print(f'[Accuracy]  : {accuracy:>2.10f}')
    accu.append(accuracy)
accu_avg = accu_sum / 9

print(f'[Accuracy_average]  : {accu_avg}, [learning rate] : {learning_rate}, [nb_epoch] : {nb_epoch}')

sub = list()
for i in range(1, 10):
    i = str(i)
    sub.append('Subject'+i)
accu.append(accu_avg)
sub.append('Average')

plt.bar(sub, accu)
plt.title(f'HS_CNN, lr = {learning_rate}, nb_epoch = {nb_epoch}')
plt.xlabel('Subject')
plt.ylabel('Accuracy')
plt.show()











######################################## 모델 기초
# train_file = np.load('.\\Post_Research\\Preprocessed_Data\\3s - 5.5s\\Beta_13-32Hz_BPF\\A01T.npz')
# test_file = np.load('.\\Post_Research\\Preprocessed_Data\\3s - 5.5s\\Beta_13-32Hz_BPF\\A01E.npz')
# beta_train = torch.FloatTensor(train_file['x'])
# y_train = torch.LongTensor(train_file['y'])
# beta_test = torch.FloatTensor(test_file['x'])
# y_test = torch.LongTensor(test_file['y'])
# train_file = np.load('.\\Post_Research\\Preprocessed_Data\\3s - 5.5s\\Theta_4-7Hz_BPF\\A01T.npz')
# test_file = np.load('.\\Post_Research\\Preprocessed_Data\\3s - 5.5s\\Theta_4-7Hz_BPF\\A01E.npz')
# theta_train = torch.FloatTensor(train_file['x'])
# theta_test = torch.FloatTensor(test_file['x'])
# train_file = np.load('.\\Post_Research\\Preprocessed_Data\\3s - 5.5s\\Mu_8-13Hz_BPF\\A01T.npz')
# test_file = np.load('.\\Post_Research\\Preprocessed_Data\\3s - 5.5s\\Mu_8-13Hz_BPF\\A01E.npz')
# mu_train = torch.FloatTensor(train_file['x'])
# mu_test = torch.FloatTensor(test_file['x'])
#
#
# ## 625 time length 208, 208, 209로 나누기
# class HS_CNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.kernel_1 = nn.Sequential(nn.Conv2d(1, 10, kernel_size=(1, 85), stride=(1, 3)),
#                                       nn.BatchNorm2d(10),
#                                       nn.Conv2d(10, 10, kernel_size=(22, 1), stride=(1, 1)),
#                                       nn.BatchNorm2d(10),
#                                       nn.ELU(),
#                                       nn.MaxPool2d(1,6))  ##10, 1, 31
#         self.kernel_2 = nn.Sequential(nn.Conv2d(1, 10, kernel_size=(1, 65), stride=(1, 3)),
#                                       nn.BatchNorm2d(10),
#                                       nn.Conv2d(10, 10, kernel_size=(22, 1), stride=(1, 1)),
#                                       nn.BatchNorm2d(10),
#                                       nn.ELU(),
#                                       nn.MaxPool2d(1,6))  ##10, 1, 32
#         self.kernel_3 = nn.Sequential(nn.Conv2d(1, 10, kernel_size=(1, 45), stride=(1, 3)),
#                                       nn.BatchNorm2d(10),
#                                       nn.Conv2d(10, 10, kernel_size=(22, 1), stride=(1, 1)),
#                                       nn.BatchNorm2d(10),
#                                       nn.ELU(),
#                                       nn.MaxPool2d(1,6))  ##10, 1, 33
#         self.linear1 = nn.Sequential(nn.Linear(2880, 100), nn.ELU(), nn.Dropout(p=0.5))
#         self.linear2 = nn.Linear(100, 4)
#
#     def forward(self, x, y, z):
#         xyz = None
#         for i in {x, y, z}:
#             x1 = self.kernel_1(i)
#             x1 = x1.view(x1.size(0), -1)
#             x2 = self.kernel_2(i)
#             x2 = x2.view(x2.size(0), -1)
#             x3 = self.kernel_3(i)
#             x3 = x3.view(x3.size(0), -1)
#             if xyz == None:
#                 xyz = torch.cat((x1, x2, x3), dim=1)    ##48 ,960
#             else:
#                 xyz = torch.cat((xyz, x1, x2, x3), dim=1)
#         x = self.linear1(xyz)  ## 48, 2880 --> 48 200
#         x = self.linear2(x)
#         return x


######################################## 모델 기초


######################################## 모델 기초 적용
# beta_train = beta_train.unsqueeze(dim=1)  ##(288, 1, 22, 625) 만들기
# theta_train = theta_train.unsqueeze(dim=1)
# mu_train = mu_train.unsqueeze(dim=1)
#
# ds_train = TensorDataset(beta_train, theta_train, mu_train, y_train)
# loader_train = DataLoader(ds_train, batch_size=48, shuffle=True)
#
# model = HS_CNN()
# loss_fn = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.01)
#
# for epoch in range(500):
#     error = None
#     for x1, x2, x3, y in loader_train:
#         model.train()
#         pred = model(x1, x2, x3)
#         loss = loss_fn(pred, y - 1)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         error = loss
#     if (epoch+1) % 50 == 0:
#         print(f'[Epoch] : {epoch+1:>5d}  [Loss] : {error:>2.10f}')
#
# with torch.no_grad():
#     beta_test = beta_test.unsqueeze(dim=1)
#     theta_test = theta_test.unsqueeze(dim=1)
#     mu_test = mu_test.unsqueeze(dim=1)
#     ds_test = TensorDataset(beta_test, theta_test, mu_test, y_test)
#     loader_test = DataLoader(ds_test, batch_size=48, shuffle=False)
#
#     correct = 0
#     for x1, x2, x3, y in loader_test:
#         model.eval()
#         pred = model(x1, x2, x3)
#         pred = torch.argmax(pred, dim=1)
#         correct += pred.eq(y - 1).sum()
# print(f'[Accuracy]  : {correct / 288:>.4f}')
######################################## 모델 기초 적용


t = time.time() - start
if t > 60:
    print(f'{t / 60:.2f} Min')
else:
    print(f'{t:.2f} Sec')
