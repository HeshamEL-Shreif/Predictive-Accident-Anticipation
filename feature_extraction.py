import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from PIL import Image
from dataloader import *



vgg16_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        VGG = models.vgg16(pretrained=True)
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
    feat_extractor = None
    if backbone == 'vgg16':
        feat_extractor = VGG16()
        feat_extractor = feat_extractor.to(device=device)
        feat_extractor.eval()
    else:
        raise NotImplementedError
    return feat_extractor

def frame_level_feature_extraction(feature_extractor, data_loader, batch_size = 32, shuffle = False):
    dataset_features = []
    dataset_labels = []
    with torch.no_grad():

     for batch in data_loader:
         videos_frames, labels = batch
         for video in videos_frames:
           frames_feat = []
           for frame in video:
             frame = vgg16_transform(frame)
             input_image = frame.unsqueeze(0)
             frame_feat = feature_extractor.forward(input_image.to(torch.device('cuda')))
             frame_feat = frame_feat.cpu()
             frame_feat = frame_feat.detach()
             torch.cuda.empty_cache()
             frames_feat.append(frame_feat)
           frames_feat = torch.stack(frames_feat)
           dataset_features.append(frames_feat)
         dataset_labels.extend(labels.tolist())
     features_dataloader = list_to_dataloader(dataset_features,
                                             dataset_labels,
                                             batch_size,
                                             shuffle)
     torch.save(features_dataloader, '/content/drive/MyDrive/dataset feature file/frame_features_dataloader.pth')
    return features_dataloader

