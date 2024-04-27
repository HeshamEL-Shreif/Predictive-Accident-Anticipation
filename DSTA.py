import torch
import torch.nn as nn
from DSA_Module import SpatialAttention
from DTA_Module import TemporalAttention
from GRU import GatedRecurrentUnit
from object_detection import object_feature
from TSAA_Module import *
from ultralytics import YOLO
from dataloader import *
from feature_extraction import init_feature_extractor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
object_detector = YOLO('yolov9e.pt')  # load an official model
object_detector = YOLO('/content/drive/MyDrive/yolov9/best.pt')
feature_extractor = init_feature_extractor(backbone='efficientnet', device=torch.device('cuda'))


class DSTA(nn.Module):
  '''
  Description:
    Dynamic Spatial Tenporal Attention (DSTA) network that learns to
    dynamically attend to the salient temporal segments and spatial regions
    of the driving scene video for the early accident anticipation.
    The network has outperformed the state-of-the-art
  args:
    device: hardware device type
    d: dimention of the feature vector
    m: numbers of frames of Dta
    input_dim: dimention of the input to gru
    n_layers: number of attention layers
    output_dim: dimention of the gru output (number of classes)
    n_objects: max number of objects to detect
  forwards:
    x: dataset video
    y: label
  return:
    outputs: predictions
    prev_states: previous states of gru
  '''
  def __init__(self, device, d, m,  input_dim, n_layers, t, output_dim, n_objects, with_tsaa=True):
        super().__init__()
        self.n_objects = n_objects
        self.d = d
        self.dsa = SpatialAttention(device, n_layers, d)
        self.dta = TemporalAttention(device, d)
        self.gru = GatedRecurrentUnit(device, input_dim, d, output_dim, n_layers)
        self.h_prev_weighted = nn.Softmax(dim=-1)(torch.randn((t, d), device=device, dtype=torch.bfloat16))
        self.h_t_prev = torch.zeros((t, d, m), device=device, dtype=torch.bfloat16)
        self.with_tsaa = with_tsaa
        self.tsaa = SelfAttentionAggregation(device, t, d)

  def forward(self, frames, labels, with_tsaa=True):
      self.with_tsaa = with_tsaa
      frames = frames.to(device)
      with torch.no_grad():
          o_t = object_feature(device=device,
                                feature_extractor=feature_extractor,
                                object_detector=object_detector,
                                frame=frames / 255,
                                n_objects=self.n_objects).to(device).to(torch.bfloat16)
          frames = nn.functional.interpolate(frames, size=(224, 224), mode='bilinear', align_corners=False)
          f_t = feature_extractor(frames).to(torch.bfloat16)

      f_t = nn.Linear(feature_extractor.dim_feat, self.d, device=device, dtype=torch.bfloat16)(f_t)
      f_t = nn.ReLU()(f_t)

      o_t = nn.Linear(feature_extractor.dim_feat, self.d, device=device, dtype=torch.bfloat16)(o_t)
      o_t = nn.ReLU()(o_t)

      alpha_t = self.dsa(o_t.permute(0, 2, 1), self.h_prev_weighted.unsqueeze(1))
      o_t_weighted = (o_t * alpha_t.unsqueeze(-1)).sum(dim=1)

      x_t = torch.cat((o_t_weighted, f_t), dim=1).to(torch.bfloat16)

      h_prev_weighted = self.h_prev_weighted.unsqueeze(0).clone().to(torch.bfloat16)
      out, h = self.gru(h_prev_weighted.to(torch.float32), x_t.unsqueeze(1).to(torch.float32))
      with torch.no_grad():
        h = h.squeeze(0)
        self.h_t_prev = self.h_t_prev.permute(0, 2, 1)
        self.h_t_prev[1, 0] = h[0]
        self.h_t_prev[2, 0:2] = h[0:2]
        self.h_t_prev[3, 0:3] = h[0:3]
        self.h_t_prev[4, 0:4] = h[0:4]
        self.h_t_prev[5, 0:5] = h[0:5]
        for i in range(6, self.h_t_prev.shape[0]):
          self.h_t_prev[i] = h[i-5:i]

        self.h_t_prev = self.h_t_prev.permute(0, 2, 1)

      beta_t = self.dta(self.h_t_prev)
      with torch.no_grad():
        self.h_prev_weighted = torch.sum((self.h_t_prev * beta_t), dim=2)

      if self.with_tsaa:
        t_pred = self.tsaa(h.to(device))
      else:
        t_pred = None

      return out, h, t_pred








