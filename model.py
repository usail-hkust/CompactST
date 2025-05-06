import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from functools import partial


class PatchifyLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.patch_length = args.patch_length
        self.patch_stride = args.patch_stride

    def forward(self, x_in):
        # [batch_size, num_patches, num_nodes, patch_length]
        out = x_in.unfold(dimension=1, size=self.patch_length, step=self.patch_stride)
        out = out.transpose(1, 2).contiguous() # [batch_size, num_nodes, num_patches, patch_length]
        return out

    
class PositionEmbed(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.add_position == "learnable":
            self.pe = nn.Parameter(torch.randn(args.num_patches, args.hidden_dim), requires_grad=True)
        else:
            pe = torch.zeros(args.num_patches, args.hidden_dim)
            position = torch.arange(0, args.num_patches).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, args.hidden_dim, 2) * -(math.log(10000.0) / args.hidden_dim))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe - pe.mean()
            pe = pe / (pe.std() * 10)
            self.pe = nn.Parameter(pe, requires_grad=False)

    def forward(self, x_hid):
        x_hid = x_hid + self.pe # [batch_size, num_nodes, num_patches, hidden_dim]
        return x_hid


class GatedAttn(nn.Module):
    def __init__(self, in_dim, out_dim, args):
        super().__init__()
        # self.input_layer = nn.Linear(in_dim, hid_dim)
        self.gate = nn.Linear(in_dim, out_dim)
        # self.output_layer = nn.Linear(hid_dim, out_dim)

    def forward(self, x_hid):
        x_hid = x_hid * F.silu(self.gate(x_hid))
        # x_hid = self.output_layer(x_hid)
        return x_hid
        

class FeedForwardLayer(nn.Module):
    def __init__(self, in_dim, out_dim, args):
        super().__init__()
        hid_dim = in_dim * args.expansion_factor
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.dropout1 = nn.Dropout(args.dropout)
        self.fc2 = nn.Linear(hid_dim, out_dim)
        self.dropout2 = nn.Dropout(args.dropout)

    def forward(self, x_hid):
        x_hid = self.dropout1(nn.functional.gelu(self.fc1(x_hid)))
        x_hid = self.fc2(x_hid)
        x_hid = self.dropout2(x_hid)
        return x_hid


class Spatial_FeedForwardLayer(nn.Module):
    def __init__(self, in_dim, out_dim, args):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.gate = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x_hid):
        x_hid = x_hid * F.silu(self.gate(x_hid))
        # x_hid = self.dropout(nn.functional.gelu(self.fc(x_hid)))
        return x_hid


class DimensionMixerBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.gated_attn = args.gated_attn
        self.norm = nn.LayerNorm(args.hidden_dim)

        self.ffn = FeedForwardLayer(args.hidden_dim, args.hidden_dim, args)

        if self.gated_attn:
            self.gating_block = GatedAttn(args.hidden_dim, args.hidden_dim, args)

    def forward(self, x_hid):
        residual = x_hid # [batch_size, num_nodes, num_patches, hidden_dim]
        x_hid = self.norm(x_hid)
        x_hid = self.ffn(x_hid) # [batch_size, num_nodes, num_patches, hidden_dim]

        if self.gated_attn:
            x_hid = self.gating_block(x_hid)

        out = x_hid + residual # [batch_size, num_nodes, num_patches, hidden_dim]
        return out    
    

class TemporalMixerBlock(nn.Module):
    def __init__(self, args, num_patches):
        super().__init__()
        self.gated_attn = args.gated_attn
        self.norm = nn.LayerNorm(args.hidden_dim)

        self.ffn = FeedForwardLayer(num_patches, num_patches, args)

        if self.gated_attn:
            self.gating_block = GatedAttn(num_patches, num_patches, args)

    def forward(self, x_hid):
        residual = x_hid # [batch_size, num_nodes, num_patches, hidden_dim]
        x_hid = self.norm(x_hid)
        
        x_hid = x_hid.transpose(2, 3) # [batch_size, num_nodes, hidden_dim, num_patches]
        x_hid = self.ffn(x_hid)

        if self.gated_attn:
            x_hid = self.gating_block(x_hid)

        x_hid = x_hid.transpose(2, 3)
        out = x_hid + residual
        return out

    
class SpatialMixerBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.gated_attn = args.gated_attn
        self.norm = nn.LayerNorm(args.hidden_dim)
        self.ffn = FeedForwardLayer(args.num_nodes, args.num_nodes, args)

        if self.gated_attn:
            self.gating_block = GatedAttn(args.num_nodes, args.num_nodes, args)

    def forward(self, x_hid, adj):
        residual = x_hid # [batch_size, num_nodes, num_patches, hidden_dim]
        x_hid = self.norm(x_hid)
        x_hid = x_hid.permute(0, 3, 2, 1) # [batch_size, hidden_dim, num_patches, num_nodes]

        if self.gated_attn:
            x_hid = self.gating_block(x_hid)
        x_hid = self.ffn(x_hid)
        
        x_hid = x_hid.permute(0, 3, 2, 1) # [batch_size, num_nodes, num_patches, hidden_dim]
        out = x_hid + residual
        return out
        

class GraphMixerBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.gated_attn = args.gated_attn
        self.norm = nn.LayerNorm(args.hidden_dim)
        self.ffn = FeedForwardLayer(args.hidden_dim, args.hidden_dim, args)

        if self.gated_attn:
            self.gating_block = GatedAttn(args.hidden_dim, args.hidden_dim, args)

    def forward(self, x_hid, adj):
        residual = x_hid # [batch_size, num_nodes, num_patches, hidden_dim]
        x_hid = self.norm(x_hid)
        
        d = adj.sum(1)
        a = adj / d.view(-1, 1)
        
        x_hid = x_hid.permute(0, 3, 2, 1) # [batch_size, hidden_dim, num_patches, num_nodes]
        x_hid = torch.einsum('bhpn,nk->bhpk', x_hid, a.t()) # [batch_size, hidden_dim, num_patches, num_nodes]
        x_hid = x_hid.permute(0, 3, 2, 1) # [batch_size, num_nodes, num_patches, hidden_dim]
        
        x_hid = self.ffn(x_hid)
        if self.gated_attn:
            x_hid = self.gating_block(x_hid)
            
        out = x_hid + residual
        return out
    
    
class MixerLayerWG(nn.Module):
    def __init__(self, args, num_patches):
        super().__init__()
        # self.spatial_mixer = SpatialMixerBlock(args)
        self.temporal_mixer = TemporalMixerBlock(args, num_patches)
        self.feature_mixer = DimensionMixerBlock(args)
        self.graph_mixer = GraphMixerBlock(args)

    def forward(self, x_hid, adj):
        batch_size, num_nodes, num_patches, hidden_dim = x_hid.shape
        # x_hid = self.spatial_mixer(x_hid, adj)
        if num_patches>1:
            x_hid = self.temporal_mixer(x_hid)
        x_hid = self.feature_mixer(x_hid)
        x_hid = self.graph_mixer(x_hid, adj)
        return x_hid

    
class MixerLayerWoG(nn.Module):
    def __init__(self, args):
        super().__init__()
        # self.graph_mixer = GraphMixerBlock(args)
        self.temporal_mixer = TemporalMixerBlock(args)
        self.feature_mixer = DimensionMixerBlock(args)

    def forward(self, x_hid, adj):
        # x_hid = self.graph_mixer(x_hid, adj)
        x_hid = self.temporal_mixer(x_hid)
        x_hid = self.feature_mixer(x_hid)
        return x_hid


class DownSample(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.hidden_dim = args.hidden_dim
        self.downsample_factor = args.downsample_factor
        self.linear = nn.Linear(self.hidden_dim * self.downsample_factor, self.hidden_dim)

    def forward(self, x_hid):
        batch_size, num_nodes, num_patches, hidden_dim = x_hid.shape
        x_hid = x_hid.view(batch_size, num_nodes, num_patches // self.downsample_factor, hidden_dim * self.downsample_factor)
        x_hid = self.linear(x_hid)
        
        return x_hid
        

class MultiscaleEncoding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.downsample_factor = args.downsample_factor
        self.mixer_layers = nn.ModuleList([MixerLayerWG(args, args.num_patches // (2**i)) for i in range(int(args.num_layers))])
        self.downsample_layers = nn.ModuleList([DownSample(args) for i in range(int(args.num_layers-1))])
        
    def forward(self, x_hid, adj):
        x_hid_all = []
        batch_size, num_nodes, num_patches, hidden_dim = x_hid.shape
        # x_hid_all.append(x_hid)

        for idx, mixer_layer in enumerate(self.mixer_layers):
            # [batch_size, num_nodes, num_patches, hidden_dim] (batch, dim, num_nodes, in_steps)
            x_hid = mixer_layer(x_hid, adj)
            x_hid_all.append(x_hid)

            # downsample
            if x_hid.shape[-2] > 1:
                x_hid = self.downsample_layers[idx](x_hid)

        return x_hid, x_hid_all


class NormExpert(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super(NormExpert, self).__init__()
        self.gamma = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        x = x * self.gamma + self.beta
        return x


class DomainNorm(nn.Module):
    def __init__(self, num_experts=16, hidden_dim=96, eps=1e-5):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([NormExpert(hidden_dim, eps) for _ in range(num_experts)])
        self.gating_network = nn.Linear(hidden_dim, num_experts, args)
        
    def forward(self, x):
        # [batch_size, in_steps, num_nodes, in_dim]
        x = x.squeeze(-1) # [batch_size, in_steps, num_nodes]
        x = x.transpose(1, 2) # [batch_size, num_nodes, in_steps]
        gate_input = x.mean(dim=1) # [batch_size, in_steps]
        gating_scores = self.gating_network(gate_input) # [batch_size, num_experts]
        top1_idx = torch.argmax(gating_scores, dim=-1)

        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1) # [batch_size, num_experts, num_nodes, in_steps]
        
        top1_output = expert_outputs[torch.arange(x.size(0)), top1_idx] # [batch_size, num_nodes, in_steps]
        top1_output = top1_output.transpose(1, 2).unsqueeze(-1)
        
        return top1_output


class NestedNorm(nn.Module):
    def __init__(self, args, eps=1e-5):
        super().__init__()
        self.num_experts = args.num_experts
        self.hidden_dim = args.in_steps
        self.experts = nn.ModuleList([NormExpert(self.hidden_dim, eps) for _ in range(self.num_experts)])
        self.gating_network = nn.Linear(self.hidden_dim, self.num_experts)
        # self.gating_network = FeedForwardLayer(self.hidden_dim, num_experts, args)

    def forward(self, x):
        # domain normalization
        mean_domain = x.mean(dim=(1, 2), keepdim=True).detach()
        x = x - mean_domain
        std_domain = torch.sqrt(torch.var(x, dim=(1, 2), keepdim=True, unbiased=False)+1e-5).detach()
        x = x/std_domain # [batch_size, in_steps, num_nodes, in_dim]

        # spatial normalization
        mean_spatial = x.mean(dim=1, keepdim=True).detach()
        x = x - mean_spatial
        std_spatial = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False)+1e-5).detach()
        x = x/std_spatial

        # [batch_size, in_steps, num_nodes, in_dim]
        x = x.squeeze(-1) # [batch_size, in_steps, num_nodes]
        x = x.transpose(1, 2) # [batch_size, num_nodes, in_steps]
        # x_fre = torch.fft.rfft(x, dim=2, norm='ortho')
        # x_fre = torch.cat([x, x_fre.real, x_fre.imag], axis=2)

        gating_scores = self.gating_network(x) # [batch_size, num_nodes, num_experts]
        gating_probs = F.softmax(gating_scores, dim=-1)
        top1_idx = torch.argmax(gating_scores, dim=-1)

        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=2) # [batch_size, num_nodes, num_experts, in_steps]
        batch_size, num_nodes, num_experts, in_steps = expert_outputs.shape
        expert_outputs = expert_outputs.reshape(batch_size * num_nodes, num_experts, in_steps) # [batch_size*num_nodes, num_experts, in_steps]
        top1_idx = top1_idx.view(-1) # [batch_size*num_nodes]
 
        gamma_buff = torch.stack([expert.gamma for expert in self.experts], dim=0)
        beta_buff = torch.stack([expert.beta for expert in self.experts], dim=0)
        gamma_buff = gamma_buff[top1_idx].reshape(batch_size, num_nodes)
        beta_buff = beta_buff[top1_idx].reshape(batch_size, num_nodes)
        gamma_buff = gamma_buff.unsqueeze(1).unsqueeze(-1)
        beta_buff = beta_buff.unsqueeze(1).unsqueeze(-1)
        
        top1_output = expert_outputs[torch.arange(batch_size * num_nodes), top1_idx] # [batch_size*num_nodes, in_steps]
        top1_output = top1_output.reshape(batch_size, num_nodes, in_steps) # [batch_size, num_nodes, in_steps]
        top1_output = top1_output.transpose(1, 2).unsqueeze(-1) # [batch_size, in_steps, num_nodes, in_dim]

        # diversity_loss = 1*(gating_probs * torch.log(gating_probs + 1e-8)).sum(dim=-1).mean()
        
        return top1_output, gamma_buff, beta_buff, mean_spatial, std_spatial, mean_domain, std_domain


class Denorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_denorm, gamma_buff, beta_buff, mean_spatial, std_spatial, mean_domain, std_domain):
        x_denorm = x_denorm - beta_buff
        x_denorm = x_denorm / (gamma_buff + 1e-5)
        
        x_denorm = x_denorm * std_spatial
        x_denorm = x_denorm + mean_spatial

        x_denorm = x_denorm * std_domain
        x_denorm = x_denorm + mean_domain
        
        return x_denorm


class STMixerEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.add_position = args.add_position
        self.patch_transform = nn.Linear(args.patch_length, args.hidden_dim)
        if self.add_position:
            self.position_emb = PositionEmbed(args)
        else:
            self.position_emb = None
            
        self.st_mixer_encoder = MultiscaleEncoding(args)

    def forward(self, x_in, adj):
        # print('self.patch_transform(x_in)', x_in.shape, '*'*20)
        x_hid = self.patch_transform(x_in) # [batch_size, num_nodes, num_patches, hidden_dim]
        if self.add_position:
            x_hid = self.position_emb(x_hid) # [batch_size, num_nodes, num_patches, hidden_dim]
        
        x_hid, x_hid_all = self.st_mixer_encoder(x_hid, adj)
        
        return x_hid, x_hid_all


class STMixerDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.out_steps = args.out_steps
        self.out_dim = args.out_dim
        self.output_layer = nn.Linear(args.hidden_dim, args.out_steps * args.out_dim)

    def forward(self, x_hid):
        batch_size, num_nodes, _, _ = x_hid.size()
        x_hid = x_hid.mean(2) # [batch_size, num_nodes, hidden_dim]
        out = self.output_layer(x_hid).view(batch_size, num_nodes, self.out_steps, self.out_dim)
        out = out.transpose(1, 2) # (batch_size, out_steps, num_nodes, output_dim)
        
        return out


class STMixerMutiDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.gated_attn = args.gated_attn
        self.out_steps = args.out_steps
        self.out_dim = args.out_dim
        self.pred_layers = nn.ModuleList([nn.Linear(args.hidden_dim, args.out_steps * args.out_dim) for i in range(args.num_layers+1)])

    def forward(self, x_hid_all):
        out_sum = []
        for idx, x_hid in enumerate(x_hid_all):
            batch_size, num_nodes, _, _ = x_hid.size()
            x_hid = x_hid.mean(2) # [batch_size, num_nodes, hidden_dim]
            # if self.gated_attn:
            #     x_hid = self.gating_blocks[idx](x_hid)
            out = self.pred_layers[idx](x_hid).view(batch_size, num_nodes, self.out_steps, self.out_dim)
            out = out.transpose(1, 2) # (batch_size, out_steps, num_nodes, output_dim)
            out_sum.append(out)
            
        out_sum = torch.stack(out_sum, dim=-1).sum(-1)
        
        return out_sum


# class STMixerMutiDecoder(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.out_steps = args.out_steps
#         self.out_dim = args.out_dim
#         self.output_layer = nn.Linear(args.hidden_dim, args.out_steps * args.out_dim)

#     def forward(self, x_hid_all):
#         out_sum = []
#         for idx, x_hid in enumerate(x_hid_all):
#             x_hid = x_hid.mean(2) # [batch_size, num_nodes, hidden_dim]
#             out_sum.append(x_hid)
#         x_hid = torch.stack(out_sum, dim=-1).sum(-1)
#         batch_size, num_nodes, _ = x_hid.size()
#         out = self.output_layer(x_hid).view(batch_size, num_nodes, self.out_steps, self.out_dim)
#         out = out.transpose(1, 2) # (batch_size, out_steps, num_nodes, output_dim)
        
#         return out


class CompactST(nn.Module):
    def __init__(self, args):
        super(CompactST, self).__init__()
        self.use_revin = args.use_revin
        self.nested_norm = NestedNorm(args)
        self.denorm = Denorm()
        
        self.patchify_layer = PatchifyLayer(args)
        self.encoder = STMixerEncoder(args)
        # self.predictor = STMixerDecoder(args)
        self.predictor = STMixerMutiDecoder(args)

    def forward(self, x_in, adj):
        # [batch_size, in_steps, num_nodes, in_dim]
        x_in = x_in[..., [0]]
        if self.use_revin:
            x_in, gamma_buff, beta_buff, mean_spatial, std_spatial, mean_domain, std_domain = self.nested_norm(x_in)

        x_in = x_in[..., 0]
        
        batch_size, in_steps, num_nodes = x_in.size()

        x_out = self.patchify_layer(x_in)
        x_out, x_out_all = self.encoder(x_out, adj) # [batch_size, num_nodes, num_patches, hidden_dim]

        x_denorm = self.predictor(x_out_all)

        if self.use_revin:
            # denormalize data
            x_denorm = self.denorm(x_denorm, gamma_buff, beta_buff, mean_spatial, std_spatial, mean_domain, std_domain)

        return x_denorm, x_out_all, [gamma_buff, beta_buff, mean_spatial, std_spatial, mean_domain, std_domain]
