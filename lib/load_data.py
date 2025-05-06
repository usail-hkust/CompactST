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

class SingleDataset(Dataset):
    def __init__(self, dataset_name, data, data_index):
        self.dataset_name = dataset_name
        self.data = data
        self.data_index = data_index
        
    def __getitem__(self, index):

        idx = list(self.data_index[index])

        data_x = self.data[idx[0]:idx[1]]
        data_y = self.data[idx[1]:idx[2]]

        return self.dataset_name, data_x, data_y

    def __len__(self):
        return len(self.data_index)

class SubsetBatchSampler(Sampler):
    def __init__(self, datasets, batch_size, shuffle):
        self.datasets = datasets
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batches_per_dataset = []
        for dataset_indices in [list(range(len(dataset))) for dataset in datasets]:
            if shuffle==True:
                random.shuffle(dataset_indices)
            batches = [dataset_indices[i:i + batch_size] for i in range(0, len(dataset_indices), batch_size)]
            self.batches_per_dataset.append(batches)
        self.dataset_offsets = list(itertools.accumulate([0] + [len(d) for d in datasets[:-1]]))

    def __iter__(self):
        all_batches = []
        for i, batches in enumerate(self.batches_per_dataset):
            offset = self.dataset_offsets[i]
            all_batches.extend([[offset + idx for idx in batch] for batch in batches])

        if self.shuffle==True:
            random.shuffle(all_batches)

        for batch in all_batches:
            yield batch

    def __len__(self):
        return sum(len(batches) for batches in self.batches_per_dataset)

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

    if args.add_time_features:
        num_times = args.time_dict[dataset_name]
        feature_list = [data] # (T, N, 1)
        T, N, _ = data.shape
        tid = [i % 24 for i in range(T)]
        tid = np.array(tid).reshape((T, 1, 1))
        time_in_day = np.tile(tid, [1, N, 1])
        feature_list.append(time_in_day)
        data = np.concatenate(feature_list, axis=-1)
        print('Auxiliary time features added: ', data.shape, ' number of time steps in one day:', num_times)
    
    # if dataset_name in args.unseen_list:
    #     train_index = all_index[:args.batch_size]
    #     val_index = all_index[:args.batch_size]
    #     test_index = all_index[int(num_samples*0.8):]
    # else:

    if dataset_name in args.dataset_list:
        train_index = all_index[:int(num_samples*args.train_ratio)]
        val_index = all_index[int(num_samples*0.7):int(num_samples*0.8)]
        test_index = all_index[int(num_samples*0.8):]
    else:
        train_index = all_index[:args.batch_size]
        val_index = all_index[:args.batch_size]
        test_index = all_index[int(num_samples*0.8):]
    
    # if dataset_name in args.dataset_list and dataset_name not in args.unseen_list:
    #     train_index = all_index[:int(num_samples*0.95)]
    #     val_index = all_index[int(num_samples*0.95):]
    #     test_index = all_index[int(num_samples*0.95):]
    # else:
    #     train_index = all_index[:int(num_samples*0.7)]
    #     val_index = all_index[int(num_samples*0.7):int(num_samples*0.8)]
    #     test_index = all_index[int(num_samples*0.8):]
    
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

    print('The number of training, validation, and testing samples: ', args.train_ratio, x_train.shape, x_val.shape, x_test.shape)

    if args.use_revin:
        scaler = None
    else:
        scaler = StandardScaler(mean=x_train[..., [0]].mean(), std=x_train[..., [0]].std())
        x_train[..., [0]] = scaler.transform(x_train[..., [0]])
        x_val[..., [0]] = scaler.transform(x_val[..., [0]])
        x_test[..., [0]] = scaler.transform(x_test[..., [0]])
    
    trainset = torch.utils.data.TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train))
    valset = torch.utils.data.TensorDataset(torch.FloatTensor(x_val), torch.FloatTensor(y_val))
    testset = torch.utils.data.TensorDataset(torch.FloatTensor(x_test), torch.FloatTensor(y_test))

    trainset_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    valset_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False)
    testset_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    return  trainset_loader, valset_loader, testset_loader, scaler

def load_data(data_list, args):
    train_data_list = []
    val_data_list = []
    test_data_list = []
    scaler_dict = {}
    adj_dict = {}

    for dataset_name in data_list:
        adj_path = args.dataset_path+'adj_mx/'+dataset_name+'.json'
        data_path = args.dataset_path+'data/'+dataset_name+'.dat'
        
        file = open(adj_path, 'r')
        dataset = json.load(file)
        data = np.memmap(data_path, dtype='float32', mode='r', shape=tuple(dataset['shape']))
        
        try:
            if dataset_name=='guangdong_air' or dataset_name=='huadong_air':
                adj_mx = torch.eye(data.shape[1])
            else:
                adj_mx = torch.FloatTensor(np.array(dataset['adj_mx']))
                diag_mx = torch.eye(adj_mx.size(0))
                adj_mx = adj_mx+diag_mx
                adj_mx = np.where(adj_mx!=0, 1, 0)
                adj_mx = torch.FloatTensor(adj_mx)
        except:
            adj_mx = torch.eye(data.shape[1])

        print(dataset_name, data.shape, adj_mx.shape, np.mean(data), np.std(data))
        
        if 'pems' in dataset_name or 'metr' in dataset_name:
            data = data[:, :, [0]]
        train_data, val_data, test_data, scaler = unify_data_format(dataset_name, data, args)
        train_data_list.append([dataset_name, train_data])
        val_data_list.append([dataset_name, val_data])
        test_data_list.append([dataset_name, test_data])
        scaler_dict[dataset_name] = scaler
        adj_dict[dataset_name] = adj_mx

    train_data_list = [(name, i) for name, data in train_data_list for i in data]
    random.seed(1024)
    random.shuffle(train_data_list)
    
    return train_data_list, val_data_list, test_data_list, scaler_dict, adj_dict


# def load_data(data_list, args):
#     train_data_list = []
#     val_data_list = []
#     test_data_list = []
#     scaler_dict = {}
#     adj_dict = {}

#     for dataset_name in data_list:
#         file_path = args.dataset_path+dataset_name+'.json'
#         file = open(file_path, 'r')
#         dataset = json.load(file)
#         data = np.array(dataset['data']) # (time_step, num_node, channel)
#         try:
#             adj_mx = torch.FloatTensor(np.array(dataset['adj_mx']))
#             diag_mx = torch.eye(adj_mx.size(0))
#             adj_mx = adj_mx+diag_mx
#             adj_mx = np.where(adj_mx!=0, 1, 0)
#             adj_mx = torch.FloatTensor(adj_mx)
#         except:
#             adj_mx = torch.eye(data.shape[1])

#         print(dataset_name, data.shape, adj_mx.shape, np.mean(data), np.std(data))
        
#         if 'pems' in dataset_name or 'metr' in dataset_name:
#             data = data[:, :, [0]]
#         train_data, val_data, test_data, scaler = unify_data_format(dataset_name, data, args)
#         train_data_list.append([dataset_name, train_data])
#         val_data_list.append([dataset_name, val_data])
#         test_data_list.append([dataset_name, test_data])
#         scaler_dict[dataset_name] = scaler
#         adj_dict[dataset_name] = adj_mx
            
#     train_data_list = [(name, i) for name, data in train_data_list for i in data]
#     random.seed(1024)
#     random.shuffle(train_data_list)
    
#     return train_data_list, val_data_list, test_data_list, scaler_dict, adj_dict
