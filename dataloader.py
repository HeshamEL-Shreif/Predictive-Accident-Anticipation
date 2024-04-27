import os
import torch
import cv2
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from torchvision import transforms
from PIL import Image

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((640, 640)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CarCrashDatasetLoader(Dataset):
    """
    description:
      custom dataloader for CCD
    args:
      data_path: dataset path (str)
      annotate_path: path of the annotate file(str)
      transform: transformation function
    returns:
      dataloader: dataloader object
    """
    def __init__(self, data_path, annote_path, transform=None):
      with torch.no_grad():
        self.transform = transform or ToTensor()
        self.data_path = data_path
        self.classes = ['Normal', 'Crash']
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}
        self.annotation_path = annote_path
        self.data_dict = self._parse_annotation_file()
        self.normal_vid_frames = [torch.tensor([1, 0]) for _ in range(50)]
        self.samples = self._make_dataset()
        print(f"Dataset with {len(self.classes)} classes: {self.classes}")

    def _make_dataset(self):
      with torch.no_grad():
        samples = []

        for class_folder in self.classes:
            class_path = os.path.join(self.data_path, class_folder)
            if not os.path.isdir(class_path):
                continue

            class_idx = self.class_to_idx[class_folder]

            for video_file in os.listdir(class_path):
                video_path = os.path.join(class_path, video_file)
                if not os.path.isfile(video_path):
                    continue
                label = torch.zeros(2) # [0, 0]
                label[class_idx] = 1
                if class_folder == 'Crash':
                    frames_label, time_to_accident = self.data_dict[video_file.replace(".mp4", "")]
                else:
                    frames_label = self.normal_vid_frames
                    time_to_accident = 0

                frames_label = torch.stack(frames_label)
                time_to_accident = torch.tensor(time_to_accident).unsqueeze(0)

                samples.append((video_path, label, frames_label, time_to_accident))
                del label
                del frames_label
                del time_to_accident

        return samples

    def _parse_annotation_file(self):
      with torch.no_grad():
        data_dict = {}
        mapping_table = str.maketrans({'[': '', ']': ''})
        with open(self.annotation_path, 'r') as f:
            for line in f:
                parts = line.strip().translate(mapping_table).split(',')
                vidname = parts[0]
                binlabels = parts[1:51]
                del parts
                binlabels = [int(x) for x in binlabels]
                index = binlabels.index(1)
                binlabels = [[1, 0] if label == 0 else [0, 1] for label in binlabels]
                binlabels_tensors = [torch.tensor(chunk) for chunk in binlabels]
                del binlabels
                data_dict[vidname] = binlabels_tensors, index
                del binlabels_tensors
                del index

        return data_dict

    def __len__(self):
        return len(self.samples)

    def _read_frames(self, video_path):
      with torch.no_grad():
        cap = cv2.VideoCapture(video_path)

        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)

            if self.transform:
                frame = self.transform(frame)
            else:
                frame = ToTensor()(frame)

            frames.append(frame)
            del frame
        cap.release()
        del cap
        if len(frames) < 50:
          fr_black = torch.zeros(3, 640, 640)
          for _ in range(50 - len(frames)):
            frames.append(fr_black)
          del fr_black

        return torch.stack(frames)

    def __getitem__(self, idx):
        video_path, label, frames_label, toa = self.samples[idx]
        frames = self._read_frames(video_path)

        return frames, label, frames_label, toa


def ccd_load_data(data_path, annote_path, transform=transform, batch_size=1, shuffle=False):
  """
  description:
    function to initiate dataloader and save dataloader to files
  args:
    data_path: path of dataset(str)
    annote_path: path of annotation file(str)
    transform: transform function(transforms)
    batch_size: batch size(int)
    shuffle: shuffle the dataset(boolean)
  return:
    data_loader:
  """

  video_dataset = CarCrashDatasetLoader(data_path,
                                      annote_path,
                                     transform=transform)
  data_loader = torch.utils.data.DataLoader(video_dataset, batch_size=batch_size, shuffle=shuffle)
  del video_dataset
  torch.save(data_loader, 'CCD_dataset_data_loader.pth')
  torch.save(data_loader, '/content/drive/MyDrive/CCD_dataset_data_loader.pth')
  return data_loader

def load_dataloader_from_files(path, batch_size = 8, transform=transform, shuffle = False):
  '''
  description:
    function to load dataloader from files
  args:
    path: path of dataloader(str)
    batch_size: batch size(int)
    transform: transform function(transforms)
    shuffle: shuffle the dataset(boolean)
  return:
    reconstructed_dataloader: dataloader
  '''
  loaded_dataset = torch.load(path)
  reconstructed_dataloader = DataLoader(loaded_dataset,
                                        batch_size=batch_size,
                                        shuffle=shuffle)
  return reconstructed_dataloader