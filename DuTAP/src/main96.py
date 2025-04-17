# -*- coding:utf-8 -*-
# HUST EPIC Maxr
import argparse
import sys
import torch
from utils.mywriter import MyWriter
from dataset import FNIRSData, ExtendFNIRSData
from train96 import train
import random
import numpy as np
from torch.utils.data import DataLoader
from model import DuTAP96
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


root_path = os.path.abspath(os.path.dirname(os.getcwd()))  #FNIRS
sys.path.append(root_path)
data_path = root_path + '/data'

def setup_seed(seed=212):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    log_dir = './log'
    os.makedirs(log_dir, exist_ok=True)
    writer = MyWriter(log_dir)
    print('------------------------------------load data------------------------------------')

    dataset_train = FNIRSData(data_path, 'train')
    dataset_test = FNIRSData(data_path, 'test')
    train_loader = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                  drop_last=True,num_workers=8)
    test_loader = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                                  drop_last=False,num_workers=8)


    print('------------------------------------train------------------------------------')
    model = DuTAP96(n_class=2, channels=53).to('cuda')
    train(model,train_loader, test_loader, writer, args, device)







if __name__ == '__main__':
    # add args
    parser = argparse.ArgumentParser(description='pofd public opinion concern prediction')
    parser.add_argument('--epochs', type=int, default=200, help='training epochs')
    parser.add_argument('--min_epochs', type=int, default=10, help='min training epochs')
    parser.add_argument('--attn_dim', type=int, default=96, help='input dimension')
    parser.add_argument('--heads', type=int, default=4, help='attention heads')
    parser.add_argument('--lr', type=float, default=0.00001, help='learning rate')
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    parser.add_argument('--use_checkpoint', type=bool, default=False, help='use checkpoint or not')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoint', help='checkpoint path')
    parser.add_argument('--checkpoint_name', type=str, default='2024-12-26_896.pth', help='model path')
    parser.add_argument('--scheduler_milestones', type=list, default=[10,20,30], help='milestones')
    parser.add_argument('--scheduler_rate', type=float, default=0.5, help='scheduler rate')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--num_classes', type=int, default=2, help='number of classes')
    args = parser.parse_args()


    setup_seed(args.seed)
    device = torch.device("cuda")
    print('cuda',torch.cuda.device_count())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main()