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


######################################## 모델 기초
train_file = np.load('.\\Preprocessed_Data\\A08T.npz')
# test_file = np.load('.\\Preprocessed_Data\\A08E.npz')
x_train = torch.Tensor(train_file['x'])
# y_train = torch.LongTensor(train_file['y'])
# x_test = torch.Tensor(test_file['x'])
# y_test = torch.LongTensor(test_file['y'])


def map(input,kernel,stride,padding):
    x = math.floor((input-kernel+2*padding)/stride+1)
    return x

print(f'세로 {map(22,5,2,0)}')
print(f'가로 {map(625,5,2,0)}')
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1,32,kernel_size=5,stride=2,padding=0),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2,2))
    def forward(self,x):
        return self.layer1(x)


# print(x_train[0].unsqueeze(dim=0).shape)
cnn = CNN()
x=x_train[0].unsqueeze(dim=0)
x=x.unsqueeze(dim=0)
print(cnn(x).shape)

######################################## 모델 기초













t = time.time() - start
if t > 60:
    print(f'{t / 60:.2f} Min')
else:
    print(f'{t:.2f} Sec')