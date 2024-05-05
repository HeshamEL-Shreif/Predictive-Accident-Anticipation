import torch
import torch.nn as nn


class SelfAttentionAggregation(nn.Module):
    """
    Description:
      Temporal Self-Attention Aggregation (TSAA): To help train the hidden layers of GRU better,
      the auxiliary module TSAA is included in the training stage only.
      The TSAA module provides learnable weights to perform a temporal self-attention aggregation
      of all the T hidden representations of each training video to predict the video class
    forwards:
      h_v: hidden representation of the video
    Parameters:
    - device (torch.device): Device.
    - t (int): Time.
    - d (int): Dimension.
    :output:
    - a_v (torch.Tensor): prediction of video class.
    """
    def __init__(self, device, t, d):
        super().__init__()
        self.w_saa =  nn.Parameter(torch.randn((1, t), device=device, requires_grad=True, dtype=torch.float32) * 0.01)
        self.dense1 = nn.Linear(d, 64, device=device, dtype=torch.bfloat16)
        self.dense2 = nn.Linear(64, 2, device=device, dtype=torch.bfloat16)

    def forward(self, h_v):
        z_v = nn.Softmax(dim=-1)(torch.matmul(h_v.T , h_v))
        z_v = torch.matmul(h_v, z_v)
        z_v = torch.matmul(self.w_saa, z_v)
        a_v = self.dense1(z_v.squeeze(0))
        a_v = nn.ReLU()(a_v)
        a_v = self.dense2(a_v)
        a_v = nn.Softmax(dim=-1)(a_v)
        return a_v