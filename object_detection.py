import torch
import torch.nn as nn
from torchvision import transforms
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from PIL import Image
from dataloader import *
from feature_extraction import *
cudnn.benchmark = True
plt.ion()

def object_feature(device, feature_extractor, object_detector, frame, n_objects=5):
  '''
  description:
    function to extract objects from frame and extract features of each object
    using the feature extractor
  args:
    device: hardware device type
    feature_extractor: feature extractor
    object_detector: object detector
    frame: frame to be processed(torch.Tensor)
    n_objects: number of objects to be extracted(int)
  return:
    cropped_regions: cropped regions of the objects(torch.Tensor)

  '''
  with torch.no_grad():
    results = object_detector.predict(frame, max_det=n_objects, verbose=False)
    cropped_regions = torch.zeros((frame.shape[0], n_objects, feature_extractor.dim_feat))
    for i, result in enumerate(results):
      boxes = result.boxes
      for j, box in enumerate(boxes.xyxy):
        x1, y1, x2, y2 = box
        cropped_region = frame[ :, int(y1.item()): int(y2.item()), int(x1.item()):int(x2.item())]
        cropped_region = cropped_region .unsqueeze(0)
        cropped_region = feature_extractor(cropped_region.to(device))
        cropped_regions[i, j] = cropped_region
  return cropped_regions
