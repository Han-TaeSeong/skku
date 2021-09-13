import scipy.io as sio  ### matlab file 변환
import os
import mne  ## EEG 처리

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch


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
        data = run[i]
        data = data[0, 0]
        #########################
        #data[0] : 22eeg&3eog ///data[1] : event///data[2] : target //len(data) : 8
        #########################
        a = data[2].flatten()
        if i == 3:
            target = a.copy()
        else:
            target = np.concatenate((target, a))
        for j in range(0, 48):
            beep = data[1].flatten()[j]
            # a = data[0].transpose()[:22, beep + 250 * 3:beep + 250 * 5].reshape(1, 22, -1)  ## 3초 ~ 5초

            #############filter
            a = data[0].transpose()[:22, beep + 250 * 3:beep + 250 * 5]  ## 3초 ~ 5초
            eeg_info = mne.create_info(ch_names=22 ,sfreq=250, ch_types='eeg')  ##   250hz
            a = mne.io.RawArray(a,info=eeg_info)
            a.filter(l_freq=0.5,h_freq=100)
            a = a.get_data().reshape(1,22,-1)
            #############filter



            if i == 3 and j == 0:
                feature = a.copy()
            else:
                feature = np.concatenate((feature, a), axis=0)

    return feature, target


a, b = get_data('.\\BCI_Competition_4_2a\\A03T.mat')
print(a)










A01T = sio.loadmat('.\\BCI_Competition_4_2a\\A01E.mat')   ## 딕셔너리 class
# print(data.shape)    ##### (1,9)
data = data.flatten()    ###(9,)
a = data[4]
b = list(a[0,0])
#

#
eeg_raw = np.array(b[0]).transpose()  ##25 X ~~
event = np.array(b[1]).transpose()
target = np.array(b[2]).transpose()

eeg_info = mne.create_info(ch_names=25 ,sfreq=250, ch_types='eeg')  ##   250hz
eeg = mne.io.RawArray(eeg_raw,info=eeg_info)



eeg.drop_channels(['22','23','24'])
eeg.pick_channels(['1','2',3])
eeg.plot(n_channels=10,scalings=100)
eeg.plot_psd(fmax=30)    ### Power Spectral Density
plt.show()