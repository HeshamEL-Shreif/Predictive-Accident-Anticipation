
import os
import torch
import cv2
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from torchvision import transforms
from PIL import Image
cudnn.benchmark = True
plt.ion()


class VideoFramesDataset(Dataset):
    def __init__(self, data_path, desired_frames=100, desired_fps=10, transform=None):
        self.desired_frames = desired_frames
        self.desired_fps = desired_fps
        self.data_path = data_path
        self.classes = sorted(os.listdir(data_path))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}
        self.samples = self._make_dataset()
        self.transform = transform or ToTensor()

    def _make_dataset(self):
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

                samples.append((video_path, class_idx))

        return samples

    def _adjust_frames_and_fps(self, frames):
        num_frames = frames.size(0)
        current_fps = num_frames / self.duration

        if current_fps < self.desired_fps:
            increase_factor = int(self.desired_fps / current_fps)
            frames = frames.repeat_interleave(increase_factor, dim=0)
        elif current_fps > self.desired_fps:
            reduction_factor = int(current_fps / self.desired_fps)
            frames = frames[::reduction_factor]

        if num_frames < self.desired_frames:
            padding_frames = torch.zeros(self.desired_frames - num_frames, *frames.size()[1:], dtype=frames.dtype)
            frames = torch.cat((frames, padding_frames))
            del padding_frames
        elif num_frames > self.desired_frames:
            frames = frames[:self.desired_frames]

        return frames

    def __len__(self):
        return len(self.samples)

    def _read_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = num_frames / cap.get(cv2.CAP_PROP_FPS)
        del num_frames

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

        return torch.stack(frames), duration

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        frames, self.duration = self._read_frames(video_path)
        frames = self._adjust_frames_and_fps(frames)

        return frames, label



transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# function that load data pt file 
def load_data(data_path, transform=transform, n_frames=100, fps=10, batch_size=1, shuffle=False):

  video_dataset = VideoFramesDataset(data_path,
                                     desired_frames=100,
                                     desired_fps=10,
                                     transform=transform)

  data_loader = torch.utils.data.DataLoader(video_dataset, batch_size=batch_size, shuffle=shuffle)
  del video_dataset
  torch.save(data_loader, 'dataset_data_loader.pth')
  return data_loader

# load dataloader object from files
def load_dataloader_from_files(path, batch_size = 1, transform=transform, shuffle = False):
  loaded_dataset = torch.load(path)
  reconstructed_dataloader = DataLoader(loaded_dataset,
                                        batch_size=batch_size,
                                        transform=transform,
                                        shuffle=shuffle)
  return reconstructed_dataloader

# converts list to dataloader 
class ListToDataSet(Dataset):
    def __init__(self, data, labels):
      self.data = data
      self.labels = labels

    def __len__(self):
      return len(self.data)

    def __getitem__(self, index):
      data_point = self.data[index]
      label = self.labels[index]
      return data_point, label


def list_to_dataloader(data, labels, batch_size, shuffle):
  dataset = ListToDataSet(data, labels)
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
  return dataloader