import torch
import torch.nn as nn
from torchvision import  models
from dataloader import *
from torchvision.models import VGG16_Weights, efficientnet_v2_m, EfficientNet_V2_M_Weights

class EfficientNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.model = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT)
    self.features = self.model.features
    self.dim_feat = 62720
  def forward(self, x):
    output = self.features(x)
    output = output.view(output.size(0), -1)
    return output

class VGG16(nn.Module):
  def __init__(self):
      super(VGG16, self).__init__()
      VGG = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
      self.feature = VGG.features
      self.classifier = nn.Sequential(*list(VGG.classifier.children())[:-3])
      pretrained_dict = VGG.state_dict()
      model_dict = self.classifier.state_dict(prefix='classifier.')
      pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
      model_dict.update(pretrained_dict)
      self.classifier.load_state_dict(model_dict, strict=False)
      self.dim_feat = 4096

  def forward(self, x):
      output = self.feature(x)
      output = output.view(output.size(0), -1)
      output = self.classifier(output)
      return output
  
  def init_feature_extractor(backbone='vgg16', device=torch.device('cuda')):
    '''
    description:
      function to initiate feature extractor
    args:
      backbone: backbone to be used(str)
      device: hardware device type(torch.device)
    return:
      feat_extractor: feature extractor
    '''
    feat_extractor = None
    if backbone == 'vgg16':
        feat_extractor = VGG16().eval()
        feat_extractor = feat_extractor.to(device=device)
        feat_extractor.eval()
    elif backbone == 'efficientnet':
        feat_extractor = EfficientNet().eval()
        feat_extractor = feat_extractor.to(device=device)
        feat_extractor.eval()
    else:
        raise NotImplementedError
    return feat_extractor