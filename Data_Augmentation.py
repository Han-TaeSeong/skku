import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



def get_aug(file):
    train_file = np.load(file)
    x_train = torch.Tensor(train_file['x'])
    y_train = torch.LongTensor(train_file['y'])


    one = None
    two = None
    three = None
    four = None

    ### class 별로 데이터 나누기
    for i in range(288):
        if y_train[i] == 1:
            if one == None:
                one = x_train[i].reshape(1, 22, 625)
            else:
                one = torch.cat((one, x_train[i].view(1, 22, 625)), dim=0)
        elif y_train[i] == 2:
            if two == None:
                two = x_train[i].reshape(1, 22, 625)
            else:
                two = torch.cat((two, x_train[i].view(1, 22, 625)), dim=0)
        elif y_train[i] == 3:
            if three == None:
                three = x_train[i].reshape(1, 22, 625)
            else:
                three = torch.cat((three, x_train[i].view(1, 22, 625)), dim=0)
        elif y_train[i] == 4:
            if four == None:
                four = x_train[i].reshape(1, 22, 625)
            else:
                four = torch.cat((four, x_train[i].view(1, 22, 625)), dim=0)

    ## Augmentation
    aug_data = None
    for k in {one, two, three, four}:
        for i in range(24):
            x = k[i]
            y = k[i + 1]
            z = k[i + 2]
            temp1 = z[:, 208:416]
            temp2 = x[:, 416:]
            x = torch.cat((x[:, :416], y[:, 416:]), dim=1).view(1,22,625)
            z = torch.cat((z[:, :208], y[:, 208:416], z[:, 416:]), dim=1).view(1,22,625)
            y = torch.cat((y[:, :208], temp1, temp2), dim=1).view(1,22,625)
            if aug_data == None:
                aug_data = torch.cat((x,y,z),dim=0)
            else:
                aug_data = torch.cat((aug_data,x,y,z),dim=0)    ##288, 22, 625

    for i in range(72):
        y_train[i]=1
        y_train[i+72]=2
        y_train[i+144]=3
        y_train[i+216]=4


    c = file.split('\\')[5]
    c = c.split('.')[0]   ##A01T
    np.savez(file='.\\Post_Research\\Preprocessed_Data\\3s - 5.5s Augmentation\\Theta_4-7Hz_BPF\\'
                  + c + '.npz', x=aug_data, y=y_train)

def make_data():
    for i in range(1, 10):
        i = str(i)
        get_aug('.\\Post_Research\\Preprocessed_Data\\3s - 5.5s\\Theta_4-7Hz_BPF\\A0' + i + 'T.npz')

make_data()