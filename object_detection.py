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
from ultralytics import YOLO

yolov8_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])


def object_level_feature_extraction(feature_extractor, object_detector, data_loader, batch_size = 32, shuffle = False, n_objects=5):
    dataset_object_feature = []
    dataset_labels = []
    with torch.no_grad():

      for batch in data_loader:
          videos_frames, labels = batch
          for video in videos_frames:
            frames_objects_feat = []
            for frame in video:
              frame = yolov8_transform(frame)
              results = object_detector(frame.unsqueeze(0), max_det=n_objects, save_crop=True)
              cropped_regions = torch.zeros((n_objects, 1, feature_extractor.dim_feat))

              for result in results:
                boxes = result.boxes
                for i, box in enumerate(boxes.xyxy):
                  x1, y1, x2, y2 = box
                  cropped_region = frame[ :, int(y1.item()): int(y2.item()), int(x1.item()):int(x2.item())]
                  cropped_region = vgg16_transform(cropped_region)
                  cropped_region = cropped_region.unsqueeze(0)
                  cropped_region = feature_extractor(cropped_region.to(torch.device('cuda')))
                  cropped_regions[i] = cropped_region
    
                frames_objects_feat.append(cropped_regions)
                del(cropped_regions)
                torch.cuda.empty_cache() 
                
            frames_objects_feat = torch.stack(frames_objects_feat)

            dataset_object_feature.append(frames_objects_feat)
            del(frames_objects_feat)
            torch.cuda.empty_cache() 
          dataset_labels.extend(labels.tolist())
      object_feature_dataloader = list_to_dataloader(dataset_object_feature,
                                                    dataset_labels,
                                                    batch_size,
                                                    shuffle)
      torch.save(object_feature_dataloader, 'object_feature_dataloader.pth')
    return object_feature_dataloader




