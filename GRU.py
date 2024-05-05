import torch
import torch.nn as nn

class  GatedRecurrentUnit(nn.Module):
  '''
  Description:
    Gated Recurrent Unit (GRU) network that learns to
    dynamically attend to the salient temporal segments and spatial regions
    of the driving scene video for the early accident anticipation.
    The network has outperformed the state-of-the-art
    At any frame t, the GRU takes the feature vector Xt and
    the hidden representation h_prev as the inputs to obtain the
    hidden representation of the current frame, ht.
  args:
    device: hardware device type
    input_dim: dimention of the input to gru
    hidden_dimention: The number of features in the hidden state h
    output_dim: dimention of the gru output (number of classes)
    n_layers: number of attention layers
  forwards:
    h_t_prev: hidden representation of the previous frame
    x_t: feature vector of the current frame
  return:
    outputs: predictions
    prev_states: previous states of gru
  '''
  def __init__(self, device, input_dim, hidden_dim, output_dim, n_layers):
      super().__init__()
      self.hidden_dim = hidden_dim
      self.n_layers = n_layers
      self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, device=device)
      self.dense1 = torch.nn.Linear(hidden_dim, 64, device=device, dtype=torch.float32)
      self.dense2 = torch.nn.Linear(64, output_dim, device=device, dtype=torch.float32)
      self.relu = nn.ReLU()

  def forward(self, h_t_prev, x_t):
      out, h = self.gru(x_t, h_t_prev)
      out = self.relu(self.dense1(out.to(torch.float32)))
      out = self.dense2(out)
      out = nn.Softmax(dim=-1)(out)
      return out, h.to(torch.float32)