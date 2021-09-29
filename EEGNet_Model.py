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


class EEGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.ZeroPad2d((62, 62, 0, 0)),
                                    nn.Conv2d(1, 8, kernel_size=(1, 125), stride=1),
                                    nn.BatchNorm2d(8),
                                    nn.Conv2d(8, 16, kernel_size=(22, 1)),
                                    nn.BatchNorm2d(16),
                                    nn.ELU(),
                                    nn.AvgPool2d(1, 4),
                                    nn.Dropout(p=0.5))
        self.layer2 = nn.Sequential(nn.ZeroPad2d((8, 8, 0, 0)),
                                    nn.Conv2d(16, 16, kernel_size=(1, 16)),
                                    nn.BatchNorm2d(16),
                                    nn.ELU(),
                                    nn.AvgPool2d(1, 8),
                                    nn.Dropout(p=0.5))
        self.flatten = nn.Linear(16 * 20, 4)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.flatten(x)
        return (x)


# class EEGNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.layer1 = nn.Sequential(nn.ZeroPad2d((32, 31, 0, 0)),
#                                     nn.Conv2d(1, 8, kernel_size=(1, 64), stride=1),
#                                     nn.BatchNorm2d(8),
#                                     nn.Conv2d(8, 16, kernel_size=(22, 1)),
#                                     nn.BatchNorm2d(16),
#                                     nn.ELU(),
#                                     nn.AvgPool2d(1, 4),
#                                     nn.Dropout(p=0.5))
#         ## 650 timepoint -> padding 62,62/kernel(1,125)
#         ## 320 timepoint -> padding 32,31/kernel(1,64)
#         self.layer2 = nn.Sequential(nn.ZeroPad2d((8, 7, 0, 0)),
#                                     nn.Conv2d(16, 16, kernel_size=(1, 16)),
#                                     nn.BatchNorm2d(16),
#                                     nn.ELU(),
#                                     nn.AvgPool2d(1, 8),
#                                     nn.Dropout(p=0.5))
#         ## 650 timepoint -> padding 8,8
#         ## 320 timepoint -> padding 8,7
#         self.flatten = nn.Linear(16 * 10, 4)
#         ## 650 timepoint ->16,20
#         ## 320 timepoint ->16,10
#
#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = x.view(x.size(0), -1)
#         x = self.flatten(x)
#         return x


accu = list()
accu_sum = 0
learning_rate = 0.005
nb_epoch = 500

for z in range(1,10):
    z = str(z)
    train_file = np.load('.\\Preprocessed_Data\\4_40Hz\\A0'+z+'T.npz')
    test_file = np.load('.\\Preprocessed_Data\\4_40Hz\\A0'+z+'E.npz')
    x_train = torch.Tensor(train_file['x'])
    y_train = torch.LongTensor(train_file['y'])
    x_test = torch.Tensor(test_file['x'])
    y_test = torch.LongTensor(test_file['y'])


    x_train = x_train.unsqueeze(dim=1)  ##(288, 1, 22, 625) 만들기
    # x_train = F.normalize(x_train, dim=3)
    ds_train = TensorDataset(x_train, y_train)
    loader_train = DataLoader(ds_train, batch_size=48, shuffle=True)

    model = EEGNet()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(nb_epoch):
        loss = 0
        for x, y in loader_train:
            model.train()
            pred = model(x)
            loss = loss_fn(pred, y - 1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 100 == 0 or epoch == (nb_epoch-1):
            print(f'Epoch : {epoch}  Loss : {loss}')
    torch.save(model, '.\\Save_Model\\4_40Hz\\EEGNet'+z+'.pth')
    # model = torch.load('.\\Save_Model\\EEGNet'+z+'.pth')

    with torch.no_grad():
        x_test = x_test.unsqueeze(dim=1)
        # x_test = F.normalize(x_test, dim=3)
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
accu_avg=accu_sum/9


print(f'Accuracy_average  : {accu_avg}, learning rate : {learning_rate}, nb_epoch : {nb_epoch}')

sub = list()
for i in range(1, 10):
    i = str(i)
    sub.append('subject'+i)
accu.append(accu_avg)
sub.append('average')

plt.bar(sub, accu)
plt.title(f'EEGNet, lr = {learning_rate}, nb_epoch = {nb_epoch}')
plt.xlabel('Subject')
plt.ylabel('Accuracy')
plt.show()



######################################## 모델 기초
# train_file = np.load('.\\Preprocessed_Data\\A06T.npz')
# test_file = np.load('.\\Preprocessed_Data\\A06E.npz')
# x_train = torch.FloatTensor(train_file['x'])
# y_train = torch.LongTensor(train_file['y'])
# x_test = torch.FloatTensor(test_file['x'])
# y_test = torch.LongTensor(test_file['y'])
#
#
# class EEGNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.layer1 = nn.Sequential(nn.ZeroPad2d((62, 62, 0, 0)),
#                                     nn.Conv2d(1, 8, kernel_size=(1, 125), stride=1),
#                                     nn.BatchNorm2d(8),
#                                     nn.Conv2d(8, 16, kernel_size=(22, 1)),
#                                     nn.BatchNorm2d(16),
#                                     nn.ELU(),
#                                     nn.AvgPool2d(1, 4),
#                                     nn.Dropout(p=0.5))
#         self.layer2 = nn.Sequential(nn.ZeroPad2d((8, 8, 0, 0)),
#                                     nn.Conv2d(16, 16, kernel_size=(1, 16)),
#                                     nn.BatchNorm2d(16),
#                                     nn.ELU(),
#                                     nn.AvgPool2d(1, 8),
#                                     nn.Dropout(p=0.5))
#         self.flatten = nn.Linear(16 * 20, 4)
#
#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = x.view(x.size(0), -1)
#         x = self.flatten(x)
#         return (x)
######################################## 모델 기초


######################################## 모델 기초 적용
# x_train = x_train.unsqueeze(dim=1)  ##(288, 1, 22, 625) 만들기
# x_train = F.normalize(x_train, dim=3)
# ds_train = TensorDataset(x_train, y_train)
# loader_train = DataLoader(ds_train, batch_size=48, shuffle=True)
#
# model = EEGNet()
# loss_fn = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.01)
#
# for epoch in range(500):
#     for x, y in loader_train:
#         model.train()
#         pred = model(x)
#         loss = loss_fn(pred, y - 1)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         if epoch % 50 == 0:
#             print(f'Epoch : {epoch}  Loss : {loss}')
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
#         pred = torch.argmax(pred, dim=1)
#         correct += pred.eq(y - 1).sum()
# print(f'Accuracy  : {correct / 288}')
######################################## 모델 기초 적용
# torch.save(model, '.\\Save_Model\\EEGNet6.pth')


t = time.time() - start
if t > 60:
    print(f'{t / 60:.2f} Min')
else:
    print(f'{t:.2f} Sec')
