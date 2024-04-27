import torch
import torch.nn as nn

class TemporalAttention(nn.Module):
  '''
  Description:
    DTA module to provide the temporal attention weights for
    aggregating the hidden representations of the most recent M frames
  args:
    device: hardware device type
    d: dimention of the feature vector
  forwards:
    h_t_prev: hidden representation of the previous frame
  return:
    beta_t: temporal attention weights
  '''
  def __init__(self, device, d):
    super().__init__()
    self.w_ta =  nn.Parameter(torch.randn((d, d), requires_grad=True, device=device, dtype=torch.bfloat16)) * 0.01
  def forward(self, h_t_prev):
    beta_t = nn.Tanh()(h_t_prev)
    beta_t = torch.matmul(self.w_ta, h_t_prev)
    beta_t = nn.Softmax(dim=-1)(beta_t)
    return beta_t

