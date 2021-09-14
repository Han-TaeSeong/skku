import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import Data_Preprocessing

savefile = np.load('.\\Preprocessed_Data\\A02T.npz')
x_train = torch.Tensor(savefile['x'])
y_train = torch.Tensor(savefile['y'])



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


model = list()
for i in range(1, 23):
    i = str(i)
    model.append('model' + i)

for i in range(0, 22):
    model[i] = DNN()
    optimizer = optim.Adam(model[i].parameters(), lr=0.01)

loss_fn = nn.CrossEntropyLoss()

for i in range(0,22):
    for epoch in range(10):
        pred = model[i]



