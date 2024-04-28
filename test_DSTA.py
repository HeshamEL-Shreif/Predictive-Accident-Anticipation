
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from Test_loop import *
from DSTA import *
plt.ion()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((640, 640)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Initialize Model

dsta = DSTA(device=torch.device('cuda'), d=512, m=5, input_dim=512+512, output_dim=2, n_layers=1, t=50, n_objects=12)
dsta.load_state_dict(torch.load('path/to/best')['dsta_state_dict'])
device = torch.device('cuda')


# Test Model

dsta.load_state_dict(torch.load('path/to/best')['dsta_state_dict'])
dsta.eval()
out = test_DSTA_model(video_path='patt/to/test/video', video_class=1,
           model=dsta, device=device, transform=transform)