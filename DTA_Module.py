import torch
import torch.nn as nn

class TemporalAttention(nn.Module):
  def __init__(self, device, d):
    super().__init__()
    self.w_ta =  nn.Parameter(torch.randn((d, d), requires_grad=True, device=device) * 0.01)
  def forward(self, h_t_prev):
    beta_t = nn.Tanh()(h_t_prev)
    beta_t = torch.matmul(self.w_ta, h_t_prev)
    beta_t = nn.Softmax()(beta_t)
    return beta_t
