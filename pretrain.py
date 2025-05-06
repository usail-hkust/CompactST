import os
import sys
import yaml
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
from torchinfo import summary

from model import CompactST
from lib.trainer import *
from lib.metrics import *
from lib.load_data import *

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
if __name__ == "__main__":
    model_size = 'stmixer_heter'
    with open('model_config/compactst.yaml', 'r') as file:
        config = yaml.safe_load(file)
    config = config[model_size]
    
    # obtain args
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=config['dataset_path'])
    parser.add_argument("--dataset_list", type=list, default=config['dataset_list'])
    parser.add_argument("--unseen_list", type=list, default=config['unseen_list'])
    parser.add_argument("--in_steps", type=int, default=config['in_steps'])
    parser.add_argument("--out_steps", type=int, default=config['out_steps'])
    parser.add_argument("--batch_size", type=int, default=config['batch_size'])
    parser.add_argument("--max_epochs", type=int, default=config['max_epochs'])
    parser.add_argument("--lr", type=float, default=config["lr"])
    parser.add_argument("--clip_grad", type=float, default=config["clip_grad"])
    parser.add_argument("--weight_decay", type=float, default=config.get("weight_decay", 0))
    parser.add_argument("--milestones", type=list, default=config["milestones"])
    parser.add_argument("--lr_decay_rate", type=float, default=config.get("lr_decay_rate", 0.1))
    parser.add_argument("--eps", type=float, default=config.get("eps", 1e-8))
    parser.add_argument("--early_stop", type=float, default=config.get("early_stop", 10))
    parser.add_argument("--log_step", type=float, default=config['log_step'])
    parser.add_argument("--use_revin", type=bool, default=config['use_revin'])
    parser.add_argument("--train_ratio", type=int, default=0.7)
    parser.add_argument("--mode", type=str, default=config.get("mode", "train"))
    parser.add_argument("--save_path", type=str, default='checkpoint/')
    args = parser.parse_args()
    
    set_seed(999)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('cuda:{}'.format(1) if torch.cuda.is_available() else "cpu")
    args.device = device
    args.patch_length = config['model_args']['patch_length']
    args.patch_stride = config['model_args']['patch_stride']
    args.in_dim = config['model_args']['in_dim']
    args.out_dim = config['model_args']['out_dim']
    args.num_patches = config['model_args']['num_patches']
    args.hidden_dim = config['model_args']['hidden_dim']
    args.num_experts = config['model_args']['num_experts']
    args.num_layers = config['model_args']['num_layers']
    args.expansion_factor = config['model_args']['expansion_factor']
    args.dropout = config['model_args']['dropout']
    args.add_position = config['model_args']['add_position']
    args.gated_attn = config['model_args']['gated_attn']
    args.num_nodes = config['model_args']['num_nodes']
    args.downsample_factor = config['model_args']['downsample_factor']
    args.add_time_features = False
    
    print(model_size, args)
    
    # load pre-training data
    trainset_loader, valset_loader, testset_loader, scaler_dict, adj_dict = load_data(args.dataset_list, args)
    _, _, unseen_loader, uscaler_dict, uadj_dict = load_data(args.unseen_list, args)
    
    # init model
    model = CompactST(args)
#     if torch.cuda.device_count() > 1:
#         model = nn.DataParallel(model)
    model = model.to(args.device)
    
    save_path = 'checkpoint/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    args.save_path = os.path.join(save_path, model_size+'.pt')
    
    loss = [MaskedMAELoss(), nn.HuberLoss()]
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay,
        eps=args.eps
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=args.milestones,
        gamma=args.lr_decay_rate, 
        verbose=False
    )
    
    print(summary(model, verbose=0))
    
    # start training
    model_trainer = trainer(args, model, trainset_loader, valset_loader, testset_loader, unseen_loader, scaler_dict, uscaler_dict,
                            adj_dict, uadj_dict, loss, optimizer, scheduler)

    print('The performance on unseen datasets before pre-training...')
    print('*'*30)
    model_trainer.test_model()
    print('*'*30)
    
    if args.mode == 'train':
        model_trainer.train_multi_epoch()
    elif args.mode == 'eval':
        pass
    elif args.mode == 'test':
        pass
    else:
        raise ValueError
    
    
    