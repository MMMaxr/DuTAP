# -*- coding:utf-8 -*-

import copy
import os
import pickle
import random
import sys

sys.path.append("../..")
#
import numpy as np
import torch
# from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
# from torch.optim.lr_scheduler import StepLR
#
# import scipy.sparse as sp
# from tqdm import tqdm
#
# from pytorchtools import EarlyStopping
# from src.models import POFD_LP, POFD_NC, POFD_DBLP
import torch.nn.functional as F
import scipy.io as sio
import os
import random
from datetime import datetime
import logging

def normalize_timewise(x):
    # x (B, C, T)
    mean = x.mean(dim=3, keepdim=True)  # (B, C, 1)
    std = x.std(dim=3, keepdim=True)  # (B, C, 1)
    x_normalized = (x - mean) / (std + 1e-6)  # 防止除以0

    return x_normalized

def generate_csv_from_mat(hb1,hb2,hb3):
    data1 = sio.loadmat(hb1)
    data2 = sio.loadmat(hb2)
    data3 = sio.loadmat(hb3)
    return data1,data2,data3



def score_to_class(score):
    if score < 9:
        return 0
    else:
        return 1


def shuffled_phrases(phrase1,phrase2,phrase3):
    original_shape = phrase1.shape
    permuted_indices = np.random.permutation(original_shape[-1])
    shuffled_phrase1 = phrase1[:, :, permuted_indices]
    shuffled_phrase2 = phrase2[:, :, permuted_indices]
    shuffled_phrase3 = phrase3[:, :, permuted_indices]
    return shuffled_phrase1,shuffled_phrase2,shuffled_phrase3

def feature_normalize(data):

    mu = np.mean(data,axis=1)

    std = np.std(data,axis=1)

    return (data - mu)/std


def time_point_shift_augment(data, shift_fraction=0.1, max_shift_fraction=0.2):
    C, T, num_features = data.shape
    num_shifts = int(T * shift_fraction)
    augmented_data = np.copy(data)
    shift_indices = np.random.choice(T, num_shifts, replace=False)
    for t in shift_indices:
        augmented_data[0:, t, :] = augmented_data[0:, t, :] * np.random.uniform(1 - max_shift_fraction,
                                                                              1 + max_shift_fraction)
    return augmented_data







