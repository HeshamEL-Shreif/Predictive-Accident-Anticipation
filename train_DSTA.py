
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from torchvision import transforms
from Test_loop import *
from dataloader import *
from feature_extraction import *
from train_val_loop import *
from DSTA import *
plt.ion()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((640, 640)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# Initiate Dataloader
data_loader = ccd_load_data (data_path = "CarCrach",
                             annote_path = "Crash-1500.txt",
                             transform=transform,
                             batch_size=45,
                             shuffle=True)

# Split the Dataset into train and validation
indices = list(range(len(data_loader.dataset)))
train_indices, val_indices = indices[:4000], indices[4000:]
train_dataset = Subset(data_loader.dataset, train_indices)
val_dataset = Subset(data_loader.dataset, val_indices)
batch_size = data_loader.batch_size
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=25, shuffle=False)
del data_loader
del train_dataset
del val_dataset
del train_indices
del val_indices
del indices

# Initialize Model, Scheduler, losses and optimizer
dsta = DSTA(device=torch.device('cuda'), d=512, m=5, input_dim=512+512, output_dim=2, n_layers=1, t=50, n_objects=12)
tsaa_loss_fn = nn.CrossEntropyLoss()
optimizer_dsta = torch.optim.Adam(params=dsta.parameters(), lr=0.0001)
scheduler_dsta = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_dsta, 'min')
device = torch.device('cuda')
num_epochs = 8
start_epoch = 0

# Initialize Model, Scheduler, losses and optimizer and continue training

dsta = DSTA(device=torch.device('cuda'), d=512, m=5, input_dim=512+512, output_dim=2, n_layers=1, t=50, n_objects=12)
dsta.load_state_dict(torch.load('path/to/best')['dsta_state_dict'])
tsaa_loss_fn = nn.CrossEntropyLoss()
optimizer_dsta = torch.optim.Adam(params=dsta.parameters(), lr=0.0001)
optimizer_dsta.load_state_dict(torch.load('path/to/best')['optimizer_dsta_state_dict'])
scheduler_dsta = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_dsta, 'min')
device = torch.device('cuda')
start_epoch = torch.load('path/to/best')['epoch']
num_epochs = 10

training_val(start_epoch=start_epoch,
         n_epochs=num_epochs,
         dsta=dsta,
         tsaa_loss_fn=tsaa_loss_fn ,
         optimizer_dsta=optimizer_dsta,
         scheduler_dsta=scheduler_dsta,
         data_loader=train_loader,
         val_data_loader=val_loader,
         device=device)
