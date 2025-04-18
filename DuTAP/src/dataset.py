from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd
import csv
from utils import tools as tools
from torch.utils.data import DataLoader
import torch
from scipy import io as sio

def load_label(id_path, mode):
    label_path = id_path.replace(mode, 'label')
    full_label = pd.read_csv(label_path, sep='\t', header=None, names=['id', 'label'])
    part_label = pd.read_csv(id_path, header=None, names=['id'])
    label = pd.merge(part_label, full_label, on='id', how='inner')
    return label


def load_hb(hb_path):  #  3*15000*53
    hb = np.zeros(15000 * 53 * 3, dtype=float).reshape(3, 15000, 53)
    f = open(hb_path, 'r', encoding='utf8')
    f_read = list(csv.reader(f))
    for i in range(53):
        hb[0, :, i] = f_read[i * 3]
        hb[1, :, i] = f_read[i * 3 + 1]
        hb[2, :, i] = f_read[i * 3 + 2]
    f.close()
    return hb

def load_hb_from_mat(data_path,pid):  # 3*3400*53
    hb = np.zeros(3400 * 53 * 3, dtype=float).reshape(3, 3400, 53)
    if pid[0]=='P':
        category='extendp'
    else:
        category='extendn'
    hb[0, :, :] = sio.loadmat(os.path.join(data_path,category,'oxy_rename',pid))['oxydata']
    hb[1, :, :] = sio.loadmat(os.path.join(data_path,category,'dxy_rename',pid))['dxydata']
    hb[2, :, :] = sio.loadmat(os.path.join(data_path,category,'total_rename',pid))['totaldata']
    return hb


class FNIRSData(Dataset):
    def __init__(self, data_path, mode):
        self.data_path = data_path
        self.hb_path = os.path.join(data_path, 'fill_nan')
        self.id_path = os.path.join(data_path, rf'{mode}.txt')
        self.label = load_label(self.id_path, mode)
        self.mode = mode

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        csv_path = os.path.join(self.hb_path, str(self.label.iloc[idx, 0]) + '_fNIRS_data.csv')
        hb = load_hb(csv_path)
        label = self.label.iloc[idx, 1]
        phrase1 = hb[:, 0:3000:, :]
        phrase2 = hb[:, 3000:9000:, :]
        phrase3 = hb[:,9000:15000:, :]

        if self.mode == 'train':
            phrase1 = tools.time_point_shift_augment(phrase1)
            phrase2 = tools.time_point_shift_augment(phrase2)
            phrase3 = tools.time_point_shift_augment(phrase3)
            shuffled_phrase1, shuffled_phrase2, shuffled_phrase3 = tools.shuffled_phrases(phrase1, phrase2, phrase3)
            return shuffled_phrase1, shuffled_phrase2, shuffled_phrase3, label,
        else:
            # return phrase1, phrase2, phrase3, label, sta, relation
            return phrase1, phrase2, phrase3, label
             # K*T*C K:3 T:15000 C:53

class ExtendFNIRSData(Dataset):
    def __init__(self, data_path, mode):
        self.data_path = data_path
        self.id_path = os.path.join(data_path, rf'{mode}.txt')
        self.label = load_label(self.id_path, mode)
        self.mode = mode

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):

        hb = load_hb_from_mat(self.data_path, str(self.label.iloc[idx, 0]))
        hb = torch.tensor(hb, dtype=torch.float32)
        label = self.label.iloc[idx, 1]
        phrase1 = hb[:, 0:680, :]
        phrase2 = hb[:, 680:2720, :]
        phrase3 = hb[:, 2720:3400, :]
        if self.mode == 'train':
            phrase1 = tools.time_point_shift_augment(phrase1)
            phrase2 = tools.time_point_shift_augment(phrase2)
            phrase3 = tools.time_point_shift_augment(phrase3)
            shuffled_phrase1, shuffled_phrase2, shuffled_phrase3 = tools.shuffled_phrases(phrase1, phrase2, phrase3)
            return phrase1, phrase2, phrase3, label,
        else:
            shuffled_phrase1, shuffled_phrase2, shuffled_phrase3 = tools.shuffled_phrases(phrase1, phrase2, phrase3)
            # return shuffled_phrase1, shuffled_phrase2, shuffled_phrase3, label
            return phrase1, phrase2, phrase3, label






if __name__ == '__main__':
    load_hb_from_mat('/home/maxinran/FNIRS/data','P_0001')
    # hb = load_hb('/home/maxinran/FNIRS/data/fill_nan/0001_fNIRS_data.csv')[0]   # 15000*53
    # hb = hb.reshape(53,15000)




#

