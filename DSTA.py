import torch
import torch.nn as nn
from DSA_Module import SpatialAttention
from DTA_Module import TemporalAttention
from GRU import GatedRecurrentUnit

class DSTA(nn.Module):
    def __init__(self, device, d, m,  input_dim, n_layers, output_dim):
        super().__init__()
        self.dsa = SpatialAttention(device, n_layers, d)
        self.dta = TemporalAttention(device, d)
        self.gru = GatedRecurrentUnit(device, input_dim, d, output_dim, n_layers)
        self.h_prev_weighted = torch.randn((4096), device=device)
        self.h_t_prev = torch.zeros((m, d), device=device)
    
    def forward(self, f_t, o_t):
        alpha_t = self.dsa(o_t, self.h_prev_weighted)
        o_t_weighted = torch.matmul(o_t, alpha_t)
        x_t = torch.cat((o_t_weighted.unsqueeze(0), f_t), dim=1)
        out, h  = self.gru(self.h_prev_weighted.unsqueeze(0), x_t)
        self.h_t_prev[1:] = self.h_t_prev[0:-1].clone()
        self.h_t_prev[0] = h
        beta_t = self.dta(self.h_t_prev.permute(1, 0))
        self.h_prev_weighted = torch.sum((self.h_t_prev.permute(1, 0) * beta_t), dim=1)
        return out










