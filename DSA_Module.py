import torch
import torch.nn as nn

class SpatialAttention(nn.Module):
  def __init__(self, device, n_layers, d):
    super().__init__()
    self.wsa = torch.randn((n_layers, d), requires_grad=True, device=device) * 0.01
    self.wg = torch.randn((d, d), requires_grad=True, device=device) * 0.01
    self.w_theta = torch.randn((d, d), requires_grad=True, device=device) * 0.01
    self.b_theta = torch.zeros((d, n_layers), requires_grad=True, device=device) * 0.01

  def forward(self, ot, h_prev_weighted):
     alpha_t = torch.matmul(self.wg, h_prev_weighted.unsqueeze(1))
     alpha_t = torch.add(alpha_t, torch.matmul(self.w_theta, ot))
     alpha_t = torch.add(alpha_t, self.b_theta)
     alpha_t = nn.Tanh()(alpha_t)
     alpha_t = torch.matmul(self.wsa, alpha_t)
     alpha_t = nn.Softmax()(alpha_t)
     alpha_t = alpha_t.squeeze(0)
     return alpha_t





