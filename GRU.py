import torch
import torch.nn as nn

class  GatedRecurrentUnit(nn.Module):
    def __init__(self, device, input_dim, hidden_dim, output_dim, n_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, device=device)
        self.dense1 = torch.nn.Linear(hidden_dim, 64)
        self.dense2 = torch.nn.Linear(64, output_dim)
        self.relu = nn.ReLU()

    def forward(self, h_t_prev, x_t):
        out, h = self.gru(x_t, h_t_prev)
        out = self.relu(self.dense1(out))
        out = self.dense2(out)
        out = nn.Softmax()(out)
        return out, h
