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
import math

start = time.time()

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=(1,10), stride=(1, 4)),
                                    nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=(1,10), stride=(1, 4)),
                                    nn.ReLU())
        self.layer3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=(1,10), stride=(1, 4)),
                                    nn.ReLU())  ##(128, 22, 19)
        self.fc1 = nn.Linear(128 * 22 * 7, 500)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.layer4 = nn.Sequential(self.fc1, nn.ReLU(), nn.Dropout(0.5))
        self.fc2 = nn.Linear(500, 4)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0),-1)
        x = self.layer4(x)
        x = self.fc2(x)
        return x

accu = list()
accu_sum = 0
for z in range(1,10):
    z = str(z)
    train_file = np.load('.\\Preprocessed_Data\\A0'+z+'T.npz')
    test_file = np.load('.\\Preprocessed_Data\\A0'+z+'E.npz')
    x_train = torch.Tensor(train_file['x'])
    y_train = torch.LongTensor(train_file['y'])
    x_test = torch.Tensor(test_file['x'])
    y_test = torch.LongTensor(test_file['y'])


    x_train = x_train.unsqueeze(dim=1)  ##(288, 1, 22, 625) 만들기
    x_train = F.normalize(x_train, dim=3)
    ds_train = TensorDataset(x_train, y_train)
    loader_train = DataLoader(ds_train, batch_size=48, shuffle=True)

    model = CNN()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(500):
        loss = 0
        for x, y in loader_train:
            model.train()
            pred = model(x)
            loss = loss_fn(pred, y - 1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 100 == 0:
            print(f'Epoch : {epoch}  Loss : {loss}')

    torch.save(model, '.\\Save_Model\\CNN' + z + '.pth')
    # model = torch.load('.\\Save_Model\\CNN'+z+'.pth')

    with torch.no_grad():
        x_test = x_test.unsqueeze(dim=1)
        x_test = F.normalize(x_test, dim=3)
        ds_test = TensorDataset(x_test, y_test)
        loader_test = DataLoader(ds_test, batch_size=48, shuffle=False)

        correct = 0
        for x, y in loader_test:
            model.eval()
            pred = model(x)
            pred = torch.argmax(pred, dim=1)
            correct += pred.eq(y - 1).sum()
    accuracy = correct/288
    accu_sum += accuracy
    print(f'Accuracy  : {accuracy}')
    accu.append(accuracy)


sub = list()
for i in range(1, 10):
    i = str(i)
    sub.append('subject'+i)
accu.append(accu_sum/9)
sub.append('average')

plt.bar(sub, accu)
plt.xlabel('Subject')
plt.ylabel('Accuracy')
plt.show()


######################################## 모델 기초
# train_file = np.load('.\\Preprocessed_Data\\A06T.npz')
# test_file = np.load('.\\Preprocessed_Data\\A06E.npz')
# x_train = torch.Tensor(train_file['x'])
# y_train = torch.LongTensor(train_file['y'])
# x_test = torch.Tensor(test_file['x'])
# y_test = torch.LongTensor(test_file['y'])
#
#
# class CNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.layer1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=(1,10), stride=(1, 4)),
#                                     nn.ReLU())
#         self.layer2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=(1,10), stride=(1, 4)),
#                                     nn.ReLU())
#         self.layer3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=(1,10), stride=(1, 4)),
#                                     nn.ReLU())  ##(128, 22, 19)
#         self.fc1 = nn.Linear(128 * 22 * 7, 500)
#         nn.init.xavier_uniform_(self.fc1.weight)
#         self.layer4 = nn.Sequential(self.fc1, nn.ReLU(), nn.Dropout(0.5))
#         self.fc2 = nn.Linear(500, 4)
#         nn.init.xavier_uniform_(self.fc2.weight)
#
#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = x.view(x.size(0),-1)
#         x = self.layer4(x)
#         x = self.fc2(x)
#         return x
# ######################################## 모델 기초
#
#
# ######################################## 모델 기초 적용
# x_train = x_train.unsqueeze(dim=1)  ##(288, 1, 22, 625) 만들기
# x_train = F.normalize(x_train, dim=3)
# ds_train = TensorDataset(x_train,y_train)
# loader_train = DataLoader(ds_train, batch_size=48, shuffle=True)
#
# model = CNN()
# loss_fn = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(),lr=0.01)
#
# for epoch in range(500):
#     for x, y in loader_train:
#         model.train()
#         pred = model(x)
#         loss = loss_fn(pred,y -1)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     if epoch%50 ==0:
#         print(f'Epoch : {epoch}  Loss : {loss}')
#
# with torch.no_grad():
#     x_test = x_test.unsqueeze(dim=1)
#     x_test = F.normalize(x_test, dim=3)
#     ds_test = TensorDataset(x_test, y_test)
#     loader_test = DataLoader(ds_test, batch_size=48, shuffle=False)
#
#     correct = 0
#     for x, y in loader_test:
#         model.eval()
#         pred = model(x)
#         pred = torch.argmax(pred,dim=1)
#         correct += pred.eq(y-1).sum()
# print(f'Accuracy  : {correct/288}')
# ######################################## 모델 기초 적용
# torch.save(model,'.\\Save_Model\\CNN6.pth')


t = time.time() - start
if t > 60:
    print(f'{t / 60:.2f} Min')
else:
    print(f'{t:.2f} Sec')
