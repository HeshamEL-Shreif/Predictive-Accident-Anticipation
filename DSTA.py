import torch
import torch.nn as nn
from DSA_Module import SpatialAttention
from DTA_Module import TemporalAttention
from GRU import GatedRecurrentUnit
from feature_extraction import vgg16_transform
from object_detection import object_feature



class DSTA(nn.Module):
    def __init__(self, device, d, m,  input_dim, n_layers, output_dim, n_objects, object_detector, feature_extractor):
        super().__init__()
        self.object_detector = object_detector
        self.feature_extractor = feature_extractor
        self.n_objects = n_objects
        self.device = device
        self.dsa = SpatialAttention(device, n_layers, d)
        self.dta = TemporalAttention(device, d)
        self.gru = GatedRecurrentUnit(device, input_dim, d, output_dim, n_layers)
        self.h_prev_weighted = torch.randn((d), device=device)
        self.h_t_prev = torch.zeros((m, d), device=device)

    def forward(self, x, y):
      outputs = []
      prev_states = []
      for frame, label in zip(x.squeeze(0), y.squeeze(0)):
        with torch.no_grad():
            o_t = object_feature(device=self.device,
                                 feature_extractor=self.feature_extractor,
                                 object_detector=self.object_detector,
                                 frame=frame,
                                 n_objects=self.n_objects).to(self.device)
            frame = vgg16_transform(frame)
            input_image = frame.unsqueeze(0)
            f_t = self.feature_extractor(input_image.to( self.device))
        alpha_t = self.dsa(o_t.T, self.h_prev_weighted)
        o_t_weighted = torch.matmul(o_t.T, alpha_t.unsqueeze(1))
        x_t = torch.cat((o_t_weighted.T, f_t), dim=1)
        h_prev_weighted = self.h_prev_weighted.unsqueeze(0).clone()
        out, h = self.gru(h_prev_weighted, x_t)
        outputs.append(out)
        prev_states.append(h)
        new_h_t_prev = torch.cat((h, self.h_t_prev[:-1]))
        self.h_t_prev = new_h_t_prev.clone()
        beta_t = self.dta(self.h_t_prev.permute(1, 0))
        self.h_prev_weighted = torch.sum((self.h_t_prev.permute(1, 0) * beta_t), dim=1)
      return torch.stack(outputs), torch.stack(prev_states)









