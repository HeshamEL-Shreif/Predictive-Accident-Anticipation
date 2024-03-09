import torch
import torch.nn as nn


class SelfAttentionAggregation(nn.Module):
    def __init__(self, device, t, d):
        super().__init__()
        self.w_saa =  nn.Parameter(torch.randn((1, t), device=device, requires_grad=True) * 0.01)
        self.dense1 = nn.Linear(d, 64, device=device)
        self.dense2 = nn.Linear(64, 2, device=device)

    def forward(self, h_v):
        z_v = nn.Softmax()(torch.matmul(h_v.T , h_v))
        z_v = torch.matmul(h_v, z_v)
        z_v = torch.matmul(self.w_saa, z_v)
        a_v = self.dense1(z_v.squeeze(0))
        a_v = nn.ReLU()(a_v)
        a_v = self.dense2(a_v)
        a_v = nn.Softmax()(a_v)
        return a_v