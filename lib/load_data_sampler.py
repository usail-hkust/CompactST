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
    
def unify_data_format(dataset_name, data, args): 
    all_index = []
    seq_length, num_nodes = data.shape
    data = data[..., np.newaxis]
    
    num_samples = seq_length - (args.in_steps+args.out_steps)+1
    
    for i in range(num_samples):
        all_index.append([i, i+args.in_steps, i+args.in_steps+args.out_steps])
    all_index = np.array(all_index)
    
    train_index = all_index[:int(num_samples*0.7)]
    val_index = all_index[int(num_samples*0.7):int(num_samples*0.8)]
    test_index = all_index[int(num_samples*0.8):]

    if args.use_revin:
        scaler = None
    else:
        print('Normalize the training data: ', data[:train_index.shape[0]+args.out_steps].mean(), \
              data[:train_index.shape[0]+args.out_steps].std())
        scaler = StandardScaler(mean=data[:train_index.shape[0]+args.out_steps].mean(), \
                                std=data[:train_index.shape[0]+args.out_steps].std())
#         scaler = MinMaxScaler(_min=data[:train_index.shape[0]+args.out_steps].min(), \
#                               _max=data[:train_index+args.out_steps.shape[0]].max())
        data = scaler.transform(data)
    
    trainset = SingleDataset(dataset_name, data, train_index)
    valset = SingleDataset(dataset_name, data, val_index)
    testset = SingleDataset(dataset_name, data, test_index)

    return  trainset, valset, testset, scaler

def load_data(args):
    train_idx_list = []
    val_idx_list = []
    test_idx_list = []
    scaler_dict = {}
    adj_dict = {}

    for dataset_name in args.dataset_list:
        file_path = args.dataset_path+dataset_name+'.json'
        file = open(file_path, 'r')
        dataset = json.load(file)
        data = np.array(dataset['data']) # (time_step, num_node, channel)
        try:
            adj_mx = torch.FloatTensor(np.array(dataset['adj_mx']))
            diag_mx = torch.eye(adj_mx.size(0))
            adj_mx = adj_mx+diag_mx
            adj_mx = np.where(adj_mx!=0, 1, 0)
            adj_mx = torch.FloatTensor(adj_mx)
        except:
            adj_mx = torch.eye(data.shape[1])

        print(dataset_name, data.shape, adj_mx.shape, np.mean(data), np.std(data))
        
        if len(data.shape) > 2:
            for i in range(data.shape[-1]):
                data_channel = data[..., i]
                new_name = dataset_name+'_'+str(i)
                train_data, val_data, test_data, scaler = unify_data_format(new_name, data_channel, args)
                train_idx_list.append(train_data)
                val_idx_list.append(val_data)
                test_idx_list.append(test_data)
                scaler_dict[new_name] = scaler
                adj_dict[new_name] = adj_mx
        else:
            train_data, val_data, test_data, scaler = unify_data_format(dataset_name, data, args)
            train_idx_list.append(train_data)
            val_idx_list.append(val_data)
            test_idx_list.append(test_data)
            scaler_dict[dataset_name] = scaler
            adj_dict[dataset_name] = adj_mx
            
    train_all = ConcatDataset(train_idx_list)
    val_all = ConcatDataset(val_idx_list)
    test_all = ConcatDataset(test_idx_list)
    
    train_sampler = SubsetBatchSampler(train_idx_list, batch_size=32, shuffle=True)
    val_sampler = SubsetBatchSampler(val_idx_list, batch_size=32, shuffle=False)
    test_sampler = SubsetBatchSampler(test_idx_list, batch_size=32, shuffle=False)
    
    train_dataloader = DataLoader(train_all, batch_sampler=train_sampler)
    val_dataloader = DataLoader(val_all, batch_sampler=val_sampler)
    test_dataloader = DataLoader(test_all, batch_sampler=test_sampler)
    
    return train_dataloader, val_dataloader, test_dataloader, scaler_dict, adj_dict
