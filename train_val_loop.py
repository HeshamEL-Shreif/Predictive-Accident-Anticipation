import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
from exp_loss_function import *
from Evaluation import *
plt.ion()

def training_val(start_epoch, n_epochs, dsta, tsaa_loss_fn, optimizer_dsta, scheduler_dsta, data_loader, val_data_loader, device):
    loss_fn = tsaa_loss_fn
    best_loss = 0.0
    best_epoch = 0
    all_recall = []
    all_precision = []
    all_ap = []
    all_mtta = []
    all_tta_r80 = []
    all_p_r80 = []
    all_train_loss = []
    all_val_loss = []


    for epoch in range(start_epoch, start_epoch + n_epochs + 1):
        #dsta.train()  # Set model to training mode
        epoch_loss = 0.0
        train_total_AP = 0.0
        train_total_mTTA = 0.0
        train_total_TTA_R80 = 0.0
        train_total_P_R80 = 0.0
        train_total_recall = 0.0
        train_total_precision = 0.0

        print(f'{datetime.datetime.now()} Epoch [{epoch}/{n_epochs}]', end=" ")

        total_batches = len(data_loader)
        count = 0
        with tqdm(total=total_batches) as pbar:
          for batch in data_loader:
              # zero gradient
              optimizer_dsta.zero_grad()

              all_frames, all_V_label, all_frames_label, all_toa = batch

              all_frames = all_frames.to(device)
              all_V_label = all_V_label.to(device)
              all_frames_label = all_frames_label.to(device)
              all_toa = all_toa.to(device)

              pred_frames = []
              pred_tssa = []
              for frames, V_label, frames_label, toa in zip(all_frames, all_V_label, all_frames_label, all_toa):
                  frames = frames.to(device)
                  V_label = V_label.to(device)
                  toa = toa.to(device)

                  if isinstance(frames_label, list):
                      frames_label = torch.tensor(frames_label).to(device)
                  else:
                      frames_label = frames_label.to(device)

                  # Forward pass
                  out, h, t_pred = dsta(frames, frames_label)
                  pred_frames.append(out)
                  pred_tssa.append(t_pred)

              #Compute Loss
              pred_frames = torch.stack(pred_frames).to(device)
              pred_tssa = torch.stack(pred_tssa).to(device)

              t, vid_class_pred = Time_to_accident_calc(pred_frames.squeeze(2) , all_V_label, all_toa, 10.0)

              frame_level_loss = exp_loss(vid_class_pred.to(device), all_V_label.to(device), time=t.to(device), toa=all_toa.squeeze(1).to(device), fps=10.0)
              loss_vid = loss_fn(pred_tssa.unsqueeze(0).to(device), all_V_label.unsqueeze(0).to(device))
              total_loss = frame_level_loss + 15 * loss_vid

              # scheduler step
              scheduler_dsta.step(total_loss.item())

              # Backward pass and update gradients for total_loss
              total_loss.backward()


              # Accumulate epoch loss
              epoch_loss += total_loss.item()

              # Update tqdm progress bar
              count+= all_V_label.shape[0]
              pbar.update(1)
              pbar.set_description(f'Training Progress | loss: {(epoch_loss / count):.4f}')


              # Update optimizer parameters over batch accumulated gradient
              optimizer_dsta.step()

        # Calculate average training loss and accuracy for the epoch
        avg_epoch_loss = epoch_loss / count

        # Save the last model checkpoint
        torch.save({
            'epoch': epoch + 1,
            'dsta_state_dict': dsta.state_dict(),
            'optimizer_dsta_state_dict': optimizer_dsta.state_dict(),
            'loss': avg_epoch_loss,
        }, f'/content/drive/MyDrive/last_checkpoint.pth')

        val_loss = 0.0
        total_AP = 0.0
        total_mTTA = 0.0
        total_TTA_R80 = 0.0
        total_P_R80 = 0.0
        total_recall = 0.0
        total_precision = 0.0
        with tqdm(total=len(val_data_loader)) as pbar2:
          with torch.no_grad():
            for val_batch in val_data_loader:
              all_pred = []
              all_frames, all_V_label, all_frames_label, all_toa = val_batch

              all_frames = all_frames.to(device)
              all_V_label = all_V_label.to(device)
              all_frames_label = all_frames_label.to(device)
              all_toa = all_toa.to(device)

              for frames, V_label, frames_label, toa in zip(all_frames, all_V_label, all_frames_label, all_toa):
                    frames = frames.to(device)
                    V_label = V_label.to(device)
                    if isinstance(frames_label, list):
                        frames_label = torch.tensor(frames_label).to(device)
                    else:
                        frames_label = frames_label.to(device)
                    toa = toa.to(device)
                    out, h, ts_pred = dsta(frames, frames_label, with_tsaa=False)
                    all_pred.append(out)


              all_pred = torch.stack(all_pred).to(device)

              t, vid_class_pred = Time_to_accident_calc(all_pred.squeeze(2) , all_V_label, all_toa, 10.0)
              loss = exp_loss(vid_class_pred.to(device), all_V_label.to(device), time=t.to(device), toa=all_toa.squeeze(1).to(device), fps=10.0)

              val_loss += loss.item()
              # Update tqdm progress bar
              pbar2.update(1)
              pbar2.set_description(f'Validation Progress')
              AP, mTTA, TTA_R80, P_R80, precision, recall = evaluation(all_pred=all_pred.squeeze(2).to(device), all_labels=all_V_label.to(device), time_of_accidents=all_toa.to(device), fps=10.0)
              total_AP += AP
              total_mTTA += mTTA
              total_TTA_R80 += TTA_R80
              total_P_R80 += P_R80
              total_recall += recall
              total_precision += precision
            avg_val_loss = val_loss / len(val_data_loader)
            avg_ap = total_AP / len(val_data_loader)
            avg_mTTA = total_mTTA / len(val_data_loader)
            avg_TTA_R80 = total_TTA_R80 / len(val_data_loader)
            avg_P_R80 = total_P_R80 / len(val_data_loader)
            avg_recall = total_recall / len(val_data_loader)
            avg_precision = total_precision / len(val_data_loader)

            output_format = """
Train Loss  Val Loss     AP       mTTA    TTA_R80     P_R80     Recall    Precision
  {:.4f}     {:.4f}   {:.4f}    {:.4f}    {:.4f}    {:.4f}     {:.4f}     {:.4f}
"""
            print(output_format.format(avg_epoch_loss, avg_val_loss, avg_ap, avg_mTTA, avg_TTA_R80, avg_P_R80, avg_recall, avg_precision))

            all_recall.append(avg_recall)
            all_precision.append(avg_precision)
            all_ap.append(avg_ap)
            all_mtta.append(avg_mTTA)
            all_tta_r80.append(avg_TTA_R80)
            all_p_r80.append(avg_P_R80)
            all_train_loss.append(avg_epoch_loss)
            all_val_loss.append(avg_val_loss)

            if epoch == start_epoch:
              best_loss = avg_val_loss

            # Save the best model checkpoint
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_epoch = epoch
                torch.save({
                    'epoch': epoch + 1,
                    'dsta_state_dict': dsta.state_dict(),
                    'optimizer_dsta_state_dict': optimizer_dsta.state_dict(),
                    'loss': avg_val_loss,
                }, f'/content/drive/MyDrive/best_checkpoint.pth')




    # After training, you can load the best model checkpoint if needed
    all_epoch = list(range(start_epoch , n_epochs + 1))
    print(f"Best model achieved at Epoch {best_epoch}, with Val Loss: {best_loss:.4f}")
    fig, axs = plt.subplots(4, 2, squeeze=False)
    plt.subplots_adjust(bottom=-2, right=0.9, left=-1, top=0.9)
    axs[0, 0].plot(all_epoch, all_train_loss, )
    axs[0, 0].set_title('Train Loss')
    axs[0, 1].plot(all_epoch, all_val_loss)
    axs[0, 1].set_title('Validation Loss')
    axs[1, 0].plot(all_epoch, all_precision)
    axs[1, 0].set_title('Precision')
    axs[1, 1].plot(all_epoch, all_recall)
    axs[1, 1].set_title('Recall')
    axs[2, 0].plot(all_epoch, all_ap)
    axs[2, 0].set_title('AP')
    axs[2, 1].plot(all_epoch, all_mtta)
    axs[2, 1].set_title('mTTA')
    axs[3, 0].plot(all_epoch, all_tta_r80)
    axs[3, 0].set_title('TTA@R80')
    axs[3, 1].plot(all_epoch, all_p_r80)
    axs[3, 1].set_title('Precision@R80')
    plt.show()



