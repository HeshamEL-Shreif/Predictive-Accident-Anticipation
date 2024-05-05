import torch
import torch.nn as nn

class SpatialAttention(nn.Module):
  '''
  Description:
      Attentions to the spatially distributed objects in a frame are unequal.
      This fact is modeled by the spatial attention weights,
      calculated using the weighted aggregation of hidden representations from the last step
  args:
    device: hardware device type
    d: dimention of the feature vector
    n_layers: number of attention layers
  forwards:
    ot: object feature vector
    h_prev_weighted: hidden representation of the previous frame
  return:
    alpha_t: spatial attention weights

  '''
  def __init__(self, device, n_layers, d):
    super().__init__()
    self.wsa =  nn.Parameter(torch.randn((d), requires_grad=True, device=device, dtype=torch.float32) * 0.01)
    self.wg =  nn.Parameter(torch.randn((d, d), requires_grad=True, device=device, dtype=torch.float32) * 0.01)
    self.w_theta =  nn.Parameter(torch.randn((d, d), requires_grad=True, device=device, dtype=torch.float32) * 0.01)
    self.b_theta =  nn.Parameter(torch.zeros((n_layers, d), requires_grad=True, device=device, dtype=torch.float32))

  def forward(self, ot, h_prev_weighted):
     alpha_t = torch.matmul(h_prev_weighted, self.wg)
     alpha_t = torch.add(alpha_t, torch.matmul(self.w_theta, ot).transpose(len(ot.shape) - 1,len(ot.shape) - 2))
     alpha_t = torch.add(alpha_t, self.b_theta)
     alpha_t = nn.Tanh()(alpha_t)
     alpha_t = torch.matmul(alpha_t, self.wsa)
     alpha_t = nn.Softmax(dim=-1)(alpha_t)
     alpha_t = alpha_t.squeeze(1)
     return alpha_t




