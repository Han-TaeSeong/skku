import scipy.io as sio   ### matlab file 변환
import os
import mne   ## EEG 처리

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch


A01T = sio.loadmat('.\\BCI_Competition_4_2a\\A01E.mat')   ## 딕셔너리 class

# print('Keys : ', A01T.keys())
# print('Values : ', A01T.values())
# print(A01T['__header__'])
# print(A01T['__version__'])
# print(A01T['__globals__'])
# print(A01T['data'])


data = A01T['data']    ###  numpy.ndarray class

# print(data.shape)    ##### (1,9)
data = data.flatten()    ###(9,)
a = data[4]
b = list(a[0,0])

# print(len(b))           ###length = 8
## b[0] length,25 EEG신호
## b[1] events
## b[2] target

eeg_raw = np.array(b[0]).transpose()  ##25 X ~~
event = np.array(b[1]).transpose()
target = np.array(b[2]).transpose()

eeg_info = mne.create_info(ch_names=25 ,sfreq=250, ch_types='eeg')  ##   250hz
eeg = mne.io.RawArray(eeg_raw,info=eeg_info)


# print(eeg)
print(eeg.time_as_index(385))
