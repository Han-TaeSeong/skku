import scipy.io as sio  ### matlab file 변환
import os
import mne  ## EEG 처리

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def get_data(file):
    MAT = sio.loadmat(file)  ## 딕셔너리 class
    #############  딕셔너리 참조
    # print('Keys : ', A01T.keys())
    # print('Values : ', A01T.values())
    # print(A01T['__header__'])
    # print(A01T['__version__'])
    # print(A01T['__globals__'])
    # print(A01T['data'])
    #############  딕셔너리 참조
    run = MAT['data']  ## array(1,9)
    run = run.flatten()  ## array(9)
    feature = 0  ## will (288,22,500) feature
    target = 0  ## will 288 target
    for i in range(3, 9):
        if file == '.\\BCI_Competition_4_2a\\A04T.mat':  ## A04T Data 결측처리
            data = run[i - 2]
        else:
            data = run[i]

        data = data[0, 0]
        #########################
        # data[0] : 22eeg&3eog ///data[1] : event///data[2] : target //len(data) : 8
        #########################
        a = data[2].flatten()
        if i == 3:
            target = a.copy()
        else:
            target = np.concatenate((target, a))

        b = data[0].transpose()
        eeg_info = mne.create_info(ch_names=25, sfreq=250, ch_types='eeg')
        b = mne.io.RawArray(b, info=eeg_info)
        b.filter(l_freq=0.5, h_freq=40)
        # b.plot(n_channels=2, scalings=100)
        # plt.show()
        b = b.get_data()
        for j in range(0, 48):
            beep = data[1].flatten()[j]  ##event
            c = b[:22, beep + 250 * 3:beep + 250 * 5 + 125].reshape(1, 22, -1)  ## 3초 ~ 5.5초

            if i == 3 and j == 0:
                feature = c.copy()
            else:
                feature = np.concatenate((feature, c), axis=0)

    c = file.split('\\')[2]
    c = c.split('.')[0]
    np.savez(file=".\\Preprocessed_Data\\" + c + '.npz', x=feature, y=target)  ## 파일저장


def make_data():
    for i in range(1, 10):
        i = str(i)
        get_data('.\\BCI_Competition_4_2a\\A0' + i + 'T.mat')
        get_data('.\\BCI_Competition_4_2a\\A0' + i + 'E.mat')


savefile = np.load('.\\Preprocessed_Data\\A02T.npz')
x = savefile['x']
y = savefile['y']


def make_channel_first():
    for k in range(1, 10):
        k = str(k)
        a = ''
        savefile = np.load('.\\Preprocessed_Data\\A0' + k + 'T.npz')
        x = savefile['x']
        y = savefile['y']
        for i in range(0, 22):
            for j in range(0, 288):
                if i == 0 and j == 0:
                    a = x[j][i][:]
                else:
                    a = np.concatenate((a, x[j][i][:]))
        x = a.reshape(22, 288, -1)

        np.savez(file=".\\Preprocessed_Data\\A0" + k + 'T_ChannelFirst.npz', x=x, y=y)

        savefile = np.load('.\\Preprocessed_Data\\A0' + k + 'E.npz')
        x = savefile['x']
        y = savefile['y']
        for i in range(0, 22):
            for j in range(0, 288):
                if i == 0 and j == 0:
                    a = x[j][i][:]
                else:
                    a = np.concatenate((a, x[j][i][:]))
        x = a.reshape(22, 288, -1)
        np.savez(file=".\\Preprocessed_Data\\A0" + k + 'E_ChannelFirst.npz', x=x, y=y)
