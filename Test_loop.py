import torch
import cv2
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from PIL import Image
from collections import deque
from IPython.display import clear_output
plt.ion()

def test_DSTA_model(video_path, video_class, model, device, transform):
   with torch.no_grad():
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)

            if transform:
                frame = transform(frame)
            else:
                frame = ToTensor()(frame)
            frames.append(frame)
            del frame
        cap.release()
        del cap

        frames = torch.stack(frames).to(device)
        out, h, t_pred = model(frames, torch.tensor([[0, 1]]), with_tsaa=False)
        out = out[:, :, 1].squeeze(1)
        pred = torch.where(out >= 0.5, out, 0)
        pred = torch.argwhere(pred)
        if pred.shape[0] > 0:
          pred = 1
        else:
          pred = 0

        out = out.to(torch.device('cpu')).to(torch.float)
        que = deque(maxlen = out.shape[0])
        for i in range(0, out.shape[0]):
          perc = out[i]
          que.append(perc)

          plt.figure(figsize=(10, 4))
          plt.axhline(y=0.5, color='r', linestyle='--', label='Threshold')
          plt.plot(que)
          plt.scatter(range(len(que)),que)

          plt.ylim(0,1)

          plt.draw()
          plt.pause(0.3)

          if i < out.shape[0] - 1:
            clear_output(wait=True)
        if video_class == 1:
          if pred == video_class:
            print("Crash detected")
          else:
            print("Crash  not detected")
        else:
          if pred == video_class:
            print("Normal Video detected")
          else:
            print("Normal Video not detected")

        return out