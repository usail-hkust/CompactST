import os
import sys
import yaml
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
from torchinfo import summary
import torch.nn.functional as F

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
    

class Spatial_FeedForwardLayer(nn.Module):
    def __init__(self, in_dim, out_dim, args):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        # self.fc2 = nn.Linear(in_dim, out_dim)
        self.dropout1 = nn.Dropout(args.dropout)
        # self.dropout2 = nn.Dropout(args.dropout)

    def forward(self, x_hid):
        x_hid = self.dropout1(nn.functional.gelu(self.fc1(x_hid)))
        # x_hid = self.fc2(x_hid)
        # x_hid = self.dropout2(x_hid)
        return x_hid
        

class SpatialMixing(nn.Module):
    def __init__(self, args):
        super().__init__()
        # self.e1 = nn.Parameter(torch.randn(args.num_nodes, 64), requires_grad=True)
        # self.e2 = nn.Parameter(torch.randn(64, args.num_nodes), requires_grad=True)
        self.e1 = nn.Parameter(torch.zeros(args.num_nodes, 32))
        self.e2 = nn.Parameter(torch.zeros(32, args.num_nodes))
        self.norm = nn.LayerNorm(args.hidden_dim)
        # self.e1= nn.Parameter(torch.empty(args.num_nodes, 64))
        # nn.init.xavier_uniform_(self.e1)
        # self.e2 = nn.Parameter(torch.empty(64, args.num_nodes))
        # nn.init.xavier_uniform_(self.e2)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x_hid):
        # residual = x_hid # [batch_size, num_nodes, num_patches, hidden_dim]
        # x_hid = self.norm(x_hid)
        
        e1 = F.relu(self.e1)
        e1 = e1 / (torch.norm(e1, p=2, dim=1).unsqueeze(-1)+1e-5)
        e2 = F.relu(self.e2)
        e2 = e2 / (torch.norm(e2, p=2, dim=1).unsqueeze(-1)+1e-5)
        # [batch_size, num_nodes, num_patches, hidden_dim]
        x_hid = torch.einsum('bnpd,kn->bkpd', x_hid, e2)
        x_hid = torch.einsum('bkpd,nk->bnpd', x_hid, e1)
        x_hid = self.dropout(nn.functional.gelu(x_hid))

        # x_hid = x_hid + residual
        return x_hid


class SpatialMixerBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.norm = nn.LayerNorm(args.hidden_dim)
        self.spatial_mixing1 = SpatialMixing(args)
        # self.spatial_mixing2 = SpatialMixing(args)

        # self.ffn = Spatial_FeedForwardLayer(args.num_nodes, args.num_nodes, args)

    def forward(self, x_hid):
        residual = x_hid # [batch_size, num_nodes, num_patches, hidden_dim]
        x_hid = self.norm(x_hid)
        x_hid = self.spatial_mixing1(x_hid)
        # x_hid = self.spatial_mixing2(x_hid)

        # x_hid = x_hid.permute(0, 3, 2, 1)
        # x_hid = self.ffn(x_hid)
        # x_hid = x_hid.permute(0, 3, 2, 1)

        out = x_hid + residual
        return out
        

class AdaptiveTune(nn.Module):
    def __init__(self, pre_trained_model, args):
        super(AdaptiveTune, self).__init__()
        self.use_revin = args.use_revin
        self.pre_trained_model = pre_trained_model
        self.hidden_dim = args.hidden_dim
        self.num_patches = args.num_patches
        self.num_nodes = args.num_nodes
        self.num_times = args.num_times
        self.out_steps = args.out_steps
        self.out_dim = args.out_dim

        if args.use_prompt:
            self.spatial_prompt1 = nn.Parameter(torch.empty(1, self.num_nodes, 1, self.hidden_dim))
            nn.init.xavier_uniform_(self.spatial_prompt1)
            self.spatial_prompt2 = nn.Parameter(torch.empty(1, self.num_nodes, 1, self.hidden_dim))
            nn.init.xavier_uniform_(self.spatial_prompt2)
        if args.use_spatial:
            self.spatial_mixer = SpatialMixerBlock(args)

        # self.domain_ffn1 = nn.Linear(64, 128)
        # self.domain_ffn2 = nn.Linear(128, 64)
        # self.dropout1 = nn.Dropout(args.dropout)
        # self.dropout2 = nn.Dropout(args.dropout)

        # self.spatial_table = nn.Parameter(torch.ones(self.num_nodes, self.spatial_dim))
        # nn.init.xavier_uniform_(self.spatial_table)
        # self.temporal_table = nn.Parameter(torch.ones(self.num_times, self.temporal_dim))
        # nn.init.xavier_uniform_(self.temporal_table)

        # self.last_size = 64
        # self.domain_ffn = nn.Linear(self.last_size+self.spatial_dim+self.temporal_dim, args.out_steps * args.out_dim)

    def forward(self, x, adj):
        x_in = x[..., [0]]
            
        _, out_all, stat = self.pre_trained_model(x_in, adj) # [batch_size, in_steps, num_nodes, in_dim]
        gamma_buff, beta_buff, mean_spatial, std_spatial, mean_domain, std_domain = stat

        out_last = out_all[-1]
        if args.use_prompt:
            out_last = self.spatial_prompt1 * out_all[-1]
            # out_last = self.spatial_prompt1 * out_all[-1] + self.spatial_prompt2 # use dual prompts
        if args.use_spatial:
            out_last = self.spatial_mixer(out_last)
        out_all[-1] = out_last
        
        batch_size, num_nodes, num_patches, hidden_dim = out_all[-1].shape
        # spatial_token = self.spatial_table.unsqueeze(0).expand(batch_size, -1, -1) # [batch_size, num_nodes, spatial_dim]
        # temporal_token = self.temporal_table[x[:, -1, :, 1].type(torch.LongTensor)] # [batch_size, num_nodes, temporal_dim]

        # out = spatial_token * out
        # out = self.dropout1(nn.functional.gelu(self.domain_ffn1(out)))
        # out = self.domain_ffn2(out)
        # out = self.dropout2(out)
        
        out = self.pre_trained_model.predictor(out_all)
        # out = torch.cat((out, spatial_token, temporal_token), dim=2) # [batch_size, num_nodes, hidden_dim]
        # out = self.domain_ffn(out).view(batch_size, num_nodes, self.out_steps, self.out_dim)
        # out = out.transpose(1, 2) # (batch_size, out_steps, num_nodes, output_dim)

        if self.use_revin:
            # denormalize data
            # print(out.shape, gamma_buff.shape, beta_buff.shape)
            out = self.pre_trained_model.denorm(out, gamma_buff, beta_buff, mean_spatial, std_spatial, mean_domain, std_domain)
        
        return out, out_all, stat

    
if __name__ == "__main__":
    model_size = 'stmixer_heter'
    with open('model_config/compactst.yaml', 'r') as file:
        config = yaml.safe_load(file)
    config = config[model_size]
    
    # obtain args
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=config['dataset_path'])
    parser.add_argument("--dataset", type=str, default='chengdu_traffic')
    parser.add_argument("--unseen", type=str, default='chengdu_traffic')
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
    parser.add_argument("--train_ratio", type=int, default=0.035)
    parser.add_argument("--mode", type=str, default=config.get("mode", "train"))
    parser.add_argument("--save_path", type=str, default=None)
    args = parser.parse_args()
    
    set_seed(999)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else "cpu")

    if 'bike' in args.dataset or 'taxi' in args.dataset:
        args.use_prompt=False
        args.use_spatial=False
    else:
        args.use_prompt=True
        args.use_spatial=True
    
    args.device = device
    args.patch_length = config['model_args']['patch_length']
    args.patch_stride = config['model_args']['patch_stride']
    args.downsample_factor = config['model_args']['downsample_factor']
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

    args.time_dict = {'chengdu_traffic': 144, 'shenzhen_traffic': 144, 'nyc_bike_inflow_10_20': 48, 'nyc_bike_outflow_10_20': 48, 
                      'nyc_taxi_inflow_10_20': 48, 'nyc_taxi_outflow_10_20': 48, 'nyc_solar_15min': 96, 'nc_solar_15min': 96, 
                      'penn_solar_15min': 96, 'nyc_solar_60min': 24, 'nc_solar_60min': 24, 'penn_solar_60min': 24, 'sd': 96,
                      'beijing_aqi': 24, 'radiation': 6}
    args.node_dict = {'chengdu_traffic': 524, 'shenzhen_traffic': 627, 'nyc_bike_inflow_10_20': 200, 'nyc_bike_outflow_10_20': 200,
                      'nyc_taxi_inflow_10_20': 200, 'nyc_taxi_outflow_10_20': 200, 'sd': 716, 'radiation': 3587, 'beijing_aqi': 35,
                      'nyc_solar_15min': 129, 'nyc_solar_60min': 129}
    args.dataset_list = [args.dataset]
    args.unseen_list = [args.unseen]
    args.add_time_features = False
    # args.num_nodes = 524
    args.num_nodes = args.node_dict[args.dataset]
    args.num_times = args.time_dict[args.dataset]
    
    print(args)
    
    # load pre-training data
    trainset_loader, valset_loader, testset_loader, scaler_dict, adj_dict = load_data(args.dataset_list, args)
    _, _, unseen_loader, uscaler_dict, uadj_dict = load_data(args.unseen_list, args)
    
    # load pre-trained model
    model = CompactST(args)
    model_path = 'checkpoint/stmixer_heter_epoch 18.pt'
    saved_state_dict = torch.load(model_path)
    model.load_state_dict(saved_state_dict)

    # model = model.to(args.device)
    
    model = AdaptiveTune(model, args)
    model = model.to(args.device)
    
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
    
    # start training with few-shot examples
    model_trainer = trainer(args, model, trainset_loader, valset_loader, testset_loader, unseen_loader, scaler_dict, uscaler_dict,
                            adj_dict, uadj_dict, loss, optimizer, scheduler)

    print('Zero-shot results...')
    print('*'*30)
    model_trainer.test_model()
    print('*'*30)
    model_trainer.train_multi_epoch()
    