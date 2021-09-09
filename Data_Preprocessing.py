from scipy import io    ### matlab file 변환
import os
import mne   ## EEG 처리

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch


A01T = io.loadmat('.\\BCI_Competition_4_2a\\A01E.mat')

print(A01T)