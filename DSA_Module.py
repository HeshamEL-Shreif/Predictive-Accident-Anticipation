import torch
import torch.nn as nn

class SpatialAttention(nn.Module):
  def __init__(self, device, n_layers, d):
    super(SpatialAttention, self).__init__()
    self.wsa = torch.randn((n_layers, d), requires_grad=True, device=device) * 0.01
    self.wg = torch.randn((d, d), requires_grad=True, device=device) * 0.01
    self.w_theta = torch.randn((d, d), requires_grad=True, device=device) * 0.01
    self.b_theta = torch.zeros((n_layers, d), requires_grad=True, device=device) * 0.01

  def forward(self, ot, h_prev_weighted):
     alpha_t = torch.matmul(h_prev_weighted, self.wg)
     alpha_t = torch.add(alpha_t, torch.matmul(ot, self.w_theta))
     alpha_t = torch.add(alpha_t, self.b_theta)
     alpha_t = nn.Tanh()(alpha_t)
     alpha_t = torch.matmul(alpha_t, self.wsa.permute(1, 0))
     alpha_t = nn.Softmax()(alpha_t)
     alpha_t = alpha_t.squeeze(1)
     return alpha_t


