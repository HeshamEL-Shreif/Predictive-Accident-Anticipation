import datetime
import torch
import torch.nn as nn
from exp_loss_function import exp_loss


import datetime

def training(n_epochs, dsta, tsaa,  optimizer_dsta, optimizer_tsaa, data_loader, device, update_grad_on_all_data=False):
    t = [i for i in range(1, 51)]
    t = torch.tensor(t).to(device)
    loss_fn = nn.CrossEntropyLoss()


    for epoch in range(1, n_epochs + 1):
        dsta.train()  # Set model to training mode
        epoch_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        if update_grad_on_all_data == True:
            optimizer_dsta.zero_grad()  # Zero the gradients
            optimizer_tsaa.zero_grad()

        for batch in data_loader:
            frames, V_label, frames_label, toa = batch
            frames = frames.to(device)
            V_label = V_label.to(device)
            if isinstance(frames_label, list):  
                frames_label = torch.tensor(frames_label).to(device)  
            else:
                frames_label = frames_label.to(device) 
            toa = toa.to(device)

            if update_grad_on_all_data == False:
                optimizer_dsta.zero_grad()  # Zero the gradients
                optimizer_tsaa.zero_grad()

            # Forward pass
            loss_frame = 0.0
            out, h = dsta(frames, frames_label)
            loss1 = exp_loss(out.squeeze(1).to(device), frames_label.squeeze(0).to(device), t, toa=toa, fps=10.0)
            loss_frame += loss1.item()
            _, pred = torch.max(out.squeeze(1).to(device), 1)
            correctss  = (pred == frames_label.squeeze(0).to(device)).sum().item()
            total_frames = frames_label.size(0)
            frame_accuracy = 100.0 * correctss / total_frames
            print(f'{datetime.datetime.now()} Epoch [{epoch}/{n_epochs}], Frames Level Loss: {loss_frame:.4f}, Frames Level Accuracy: {frame_accuracy:.4f}')

            t_pred = tsaa(h.squeeze(1).to(device))
            loss_vid = loss_fn(t_pred.unsqueeze(0), V_label.to(device))
            total_loss = loss1 + 15 * loss_vid

            # Backward pass and update gradients for total_loss
            total_loss.backward()

            # Update optimizer parameters after the backward pass
            if update_grad_on_all_data == False:
                optimizer_dsta.step()
                optimizer_tsaa.step()

            # Calculate training accuracy
            _, predicted = torch.max(t_pred.data, 1)
            correct_predictions += (predicted == V_label).sum().item()
            total_samples += V_label.size(0)

            # Accumulate epoch loss
            epoch_loss += total_loss.item()

        if update_grad_on_all_data == True:
            optimizer_dsta.step()
            optimizer_tsaa.step()
    # Calculate average training loss and accuracy for the epoch
        avg_epoch_loss = epoch_loss / len(data_loader)
        accuracy = correct_predictions / total_samples

        print(f'{datetime.datetime.now()} Epoch [{epoch}/{n_epochs}], Training Loss: {avg_epoch_loss:.4f}, Training Accuracy: {accuracy:.4f}')

def validation(dsta, tsaa, data_loader, device):
    t = [i for i in range(1, 51)]
    t = torch.tensor(t).to(device)
    dsta.eval()  # Set model to evaluation mode
    total_samples = 0
    correct_predictions = 0
    epoch_loss = 0.0
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0

    with torch.no_grad():  # Disable gradient calculation for validation
        for batch in data_loader:
            frames, V_label, frames_label, toa = batch
            frames = frames.to(device)
            V_label = V_label.to(device)
            if isinstance(frames_label, list):  
                frames_label = torch.tensor(frames_label).to(device)  
            else:
                frames_label = frames_label.to(device) 
            toa = toa.to(device)

            # Forward pass
            out, h = dsta(frames, frames_label)
            loss1 = exp_loss(out.squeeze(1).to(device), frames_label.squeeze(0).to(device), t, toa=toa, fps=10.0)
            loss_frame = loss1.item()
            _, pred = torch.max(out.squeeze(1).to(device), 1)
            correctss  = (pred == frames_label.squeeze(0).to(device)).sum().item()
            total_frames = frames_label.size(0)
            frame_accuracy = 100.0 * correctss / total_frames
            print(f'{datetime.datetime.now()}  Frames Level Loss: {loss_frame:.4f}, Frames Level Accuracy: {frame_accuracy:.4f}')

            total_loss += loss_frame
            correct_predictions += correctss
            total_samples += total_frames

    # Calculate average validation loss and accuracy
    avg_epoch_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / total_samples

    print(f'Validation Total Loss: {avg_epoch_loss:.4f}, Validation Total Accuracy: {accuracy:.4f}')

# Training loop with validation
def train_with_validation(n_epochs, dsta, tsaa, optimizer_dsta, optimizer_tsaa, train_loader, val_loader, device):
    t = [i for i in range(1, 51)]
    t = torch.tensor(t).to(device)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, n_epochs + 1):
        dsta.train()  # Set model to training mode
        epoch_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        optimizer_dsta.zero_grad()  # Zero the gradients
        optimizer_tsaa.zero_grad()

        for batch in train_loader:
            frames, V_label, frames_label, toa = batch
            frames = frames.to(device)
            V_label = V_label.to(device)
            if isinstance(frames_label, list):  
                frames_label = torch.tensor(frames_label).to(device)  
            else:
                frames_label = frames_label.to(device) 
            toa = toa.to(device)

            # Forward pass
            loss_frame = 0.0
            out, h = dsta(frames, frames_label)
            loss1 = exp_loss(out.squeeze(1).to(device), frames_label.squeeze(0).to(device), t, toa=toa, fps=10.0)
            loss_frame += loss1.item()
            _, pred = torch.max(out.squeeze(1).to(device), 1)
            correctss  = (pred == frames_label.squeeze(0).to(device)).sum().item()
            total_frames = frames_label.size(0)
            frame_accuracy = 100.0 * correctss / total_frames

            t_pred = tsaa(h.squeeze(1).to(device))
            loss_vid = loss_fn(t_pred.unsqueeze(0), V_label.to(device))
            total_loss = loss1 + 15 * loss_vid

            # Backward pass and update gradients for total_loss
            total_loss.backward()

            # Update optimizer parameters after the backward pass
            optimizer_dsta.step()
            optimizer_tsaa.step()

            # Calculate training accuracy
            _, predicted = torch.max(t_pred.data, 1)
            correct_predictions += (predicted == V_label).sum().item()
            total_samples += V_label.size(0)

            # Accumulate epoch loss
            epoch_loss += total_loss.item()

        # Calculate average training loss and accuracy for the epoch
        avg_epoch_loss = epoch_loss / len(train_loader)
        accuracy = correct_predictions / total_samples

        print(f'{datetime.datetime.now()} Epoch [{epoch}/{n_epochs}], Training Loss: {avg_epoch_loss:.4f}, Training Accuracy: {accuracy:.4f}')

        # Run validation
        print("Validation:")
        validation(dsta, tsaa, val_loader, device)


