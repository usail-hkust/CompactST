import os
import json
import torch
import random
import itertools
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Sampler

class MinMaxScaler:
    def __init__(self, _min=None, _max=None):
        self._min = _min
        self._max = _max

    def transform(self, data):
        data = 1. * (data - self._min) / (self._max - self._min)
        data = data * 2. - 1.
        return data

    def inverse_transform(self, data):
        data = (data + 1.) / 2.
        data = 1. * data * (self._max - self._min) + self._min
        return data

class StandardScaler:
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit_transform(self, data):
        self.mean = data.mean()
        self.std = data.std()

        return (data - self.mean) / self.std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def vrange(starts, stops):
    stops = np.asarray(stops)
    l = stops - starts  # Lengths of each range. Should be equal, e.g. [12, 12, 12, ...]
    assert l.min() == l.max(), "Lengths of each range should be equal."
    indices = np.repeat(stops - l.cumsum(), l) + np.arange(l.sum())
    return indices.reshape(-1, l[0])    
    
def unify_data_format(dataset_name, data, args): 
    all_index = []
    # data = data[..., np.newaxis]
    seq_length, num_nodes, feat_dim = data.shape
    
    num_samples = seq_length - (args.in_steps+args.out_steps)+1
    
    for i in range(num_samples):
        all_index.append([i, i+args.in_steps, i+args.in_steps+args.out_steps])
    all_index = np.array(all_index)

    train_index = all_index[:int(num_samples*0.6)]
    val_index = all_index[int(num_samples*0.6):int(num_samples*0.8)]
    test_index = all_index[int(num_samples*0.8):]
    
    x_train_index = vrange(train_index[:, 0], train_index[:, 1])
    y_train_index = vrange(train_index[:, 1], train_index[:, 2])
    x_val_index = vrange(val_index[:, 0], val_index[:, 1])
    y_val_index = vrange(val_index[:, 1], val_index[:, 2])
    x_test_index = vrange(test_index[:, 0], test_index[:, 1])
    y_test_index = vrange(test_index[:, 1], test_index[:, 2])
    
    x_train = data[x_train_index]
    y_train = data[y_train_index][..., [0]]
    x_val = data[x_val_index]
    y_val = data[y_val_index][..., [0]]
    x_test = data[x_test_index]
    y_test = data[y_test_index][..., [0]]

    scaler = StandardScaler(mean=x_train[..., 0].mean(), std=x_train[..., 0].std())
    # x_train[..., 0] = scaler.transform(x_train[..., 0])
    # x_val[..., 0] = scaler.transform(x_val[..., 0])
    # x_test[..., 0] = scaler.transform(x_test[..., 0])

    print('The number of training samples: ', x_train.shape, x_val.shape, x_test.shape, scaler.mean, scaler.std)
    
    trainset = torch.utils.data.TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train))
    valset = torch.utils.data.TensorDataset(torch.FloatTensor(x_val), torch.FloatTensor(y_val))
    testset = torch.utils.data.TensorDataset(torch.FloatTensor(x_test), torch.FloatTensor(y_test))

    trainset_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    valset_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False)
    testset_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    return  trainset_loader, valset_loader, testset_loader, scaler

def load_data(dataset_name, args):
    train_data_list = []
    val_data_list = []
    test_data_list = []
    scaler_dict = {}
    adj_dict = {}
    
    adj_path = args.dataset_path+'adj_mx/'+dataset_name+'.json'
    data_path = args.dataset_path+'data/'+dataset_name+'.dat'
    
    file = open(adj_path, 'r')
    dataset = json.load(file)
    data = np.memmap(data_path, dtype='float32', mode='r', shape=tuple(dataset['shape']))
    
    try:
        adj_mx = torch.FloatTensor(np.array(dataset['adj_mx']))
        diag_mx = torch.eye(adj_mx.size(0))
        adj_mx = adj_mx+diag_mx
        adj_mx = np.where(adj_mx!=0, 1, 0)
        adj_mx = torch.FloatTensor(adj_mx)
    except:
        adj_mx = torch.eye(data.shape[1])
    
    # if 'pems' in dataset_name or 'metr' in dataset_name:
    #     data = data[..., [0]]
    print(dataset_name, data.shape, adj_mx.shape, np.mean(data), np.std(data))
    
    train_data, val_data, test_data, scaler = unify_data_format(dataset_name, data, args)

    train_data = [i for i in train_data]
    random.seed(1024)
    random.shuffle(train_data_list)
    
    return train_data, val_data, test_data, scaler, adj_mx
