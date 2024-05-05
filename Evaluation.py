import torch
import torch.nn as nn
import numpy as np

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def evaluation(all_pred, all_labels, time_of_accidents, fps=20.0):
    """
    Description:
      The evaluation function for the accident anticipation task.
    Parameters:
    - all_pred (torch.Tensor): Predictions (N x T x  2).
    - all_labels (torch.Tensor): Labels (N x 2).
    - time_of_accidents (torch.Tensor): Time of accidents (N x 1).
    - fps (float): Frames per second.
    output:
      AP (average precision, AUC),
      mTTA (mean Time-to-Accident),
      TTA@R80 (TTA at Recall=80%)
      P_R80 (Precision at Recall=80%)
      Recall
      Precision
    """
    all_pred = all_pred[:, :, 1].to(torch.device("cpu"))
    all_labels = all_labels[:, 1].to(torch.device("cpu"))
    time_of_accidents = time_of_accidents.squeeze(1).to(torch.device("cpu"))
    all_pred = torch.Tensor.numpy(all_pred)
    all_labels = torch.Tensor.numpy(all_labels)
    time_of_accidents = torch.Tensor.numpy(time_of_accidents)

    preds_eval = []
    min_pred = np.inf
    n_frames = 0
    for idx, toa in enumerate(time_of_accidents):
        if all_labels[idx] > 0:
            pred = all_pred[idx, :int(toa)]  # positive video
        else:
            pred = all_pred[idx, :]  # negative video
        # find the minimum prediction
        min_pred = np.min(pred) if min_pred > np.min(pred) else min_pred
        preds_eval.append(pred)
        n_frames += len(pred)
    total_seconds = all_pred.shape[1] / fps

    # iterate a set of thresholds from the minimum predictions
    temp_shape = int((1.0 - max(min_pred, 0)) / 0.001 + 0.5)
    '''
    Precision = np.zeros((n_frames))
    Recall = np.zeros((n_frames))
    Time = np.zeros((n_frames))
    '''
    Precision = []
    Recall = []
    Time = []
    cnt = 0
    for Th in np.arange(max(min_pred, 0), 1.0, 0.01):
        Tp = 0.0
        Tp_Fp = 0.0
        Tp_Tn = 0.0
        time = 0.0
        counter = 0.0  # number of TP videos
        # iterate each video sample
        for i in range(len(preds_eval)):
            # true positive frames: (pred->1) * (gt->1)
            tp =  np.where(preds_eval[i]*all_labels[i]>=Th)
            Tp += float(len(tp[0])>0)
            if float(len(tp[0])>0) > 0:
                # if at least one TP, compute the relative (1 - rTTA)
                time += tp[0][0] / float(time_of_accidents[i])
                counter = counter+1
            # all positive frames
            Tp_Fp += float(len(np.where(preds_eval[i]>=Th)[0])>0)
        if Tp_Fp == 0:  # predictions of all videos are negative
            continue
        else:
            Precision.append(Tp/Tp_Fp)
        if np.sum(all_labels) == 0: # gt of all videos are negative
            continue
        else:
            Recall.append(Tp/np.sum(all_labels))
        if counter == 0:
            Time.append(0)
            continue
        else:
            Time.append(1-time/counter)

        cnt += 1
    Precision = np.array(Precision)
    Recall  = np.array(Recall)
    Time  = np.array(Time)
    # sort the metrics with recall (ascending)
    new_index = np.argsort(Recall)
    Precision = Precision[new_index]
    Recall = Recall[new_index]
    Time = Time[new_index]
    # unique the recall, and fetch corresponding precisions and TTAs
    _,rep_index = np.unique(Recall,return_index=1)
    rep_index = rep_index[1:]
    new_Time = np.zeros(len(rep_index))
    new_Precision = np.zeros(len(rep_index))
    for i in range(len(rep_index)-1):
          new_Time[i] = np.max(Time[rep_index[i]:rep_index[i+1]])
          new_Precision[i] = np.max(Precision[rep_index[i]:rep_index[i+1]])
    # sort by descending order
    new_Time[-1] = Time[rep_index[-1]]
    new_Precision[-1] = Precision[rep_index[-1]]
    new_Recall = Recall[rep_index]
    # compute AP (area under P-R curve)
    AP = 0.0
    if new_Recall[0] != 0:
        AP += new_Precision[0]*(new_Recall[0]-0)
    for i in range(1,len(new_Precision)):
        AP += (new_Precision[i-1]+new_Precision[i])*(new_Recall[i]-new_Recall[i-1])/2

    # transform the relative mTTA to seconds
    mTTA = np.mean(new_Time) * total_seconds
    sort_time = new_Time[np.argsort(new_Recall)]
    sort_recall = np.sort(new_Recall)
    a = np.where(new_Recall>=0.8)
    P_R80 = new_Precision[a[0][0]]
    TTA_R80 = sort_time[np.argmin(np.abs(sort_recall-0.8))] * total_seconds


    return AP, mTTA, TTA_R80, P_R80, np.mean(Precision), np.mean(Recall)


def accident_anticipation_evaluation(all_pred, all_labels, time_of_accidents, fps):
    """
    Description:
      The evaluation function for the accident anticipation task.
    Parameters:
    - all_pred (torch.Tensor): Predictions (N x T x  2).
    - all_labels (torch.Tensor): Labels (N x 2).
    - time_of_accidents (torch.Tensor): Time of accidents (N x 1).
    - fps (float): Frames per second.
    output:
      AP (average precision, AUC),
      mTTA (mean Time-to-Accident),
      TTA@R80 (TTA at Recall=80%)
      P_R80 (Precision at Recall=80%)
      Recall
      Precision
    """
    with torch.no_grad():
      all_pred = all_pred[:, :, 1].to(device)
      all_labels = all_labels[:, 1].to(device)
      time_of_accidents = time_of_accidents.squeeze(1).to(device)

      min_pred = torch.inf
      min_pred = min(min_pred, torch.min(all_pred).item())

      precision = []
      recall = []
      all_time = []

      for threshold in torch.arange(max(min_pred, 0), 1.0, 0.01):
        tp = 0.0
        fp = 0.0
        fn = 0.0
        time = []
        vid_pred = torch.zeros(all_labels.shape[0]).to(device)
        for idx, toa in enumerate(time_of_accidents):
          if all_labels[idx] == 1:
            t = all_pred[idx, :toa]
            t = torch.where(t >= threshold, t, 0.0)
            t = torch.argwhere(t)
            predict = all_pred[idx, :toa]
            if t.shape[0] > 0:
              t = t[0]
              time.append((toa - (t + 1) ) / fps)
          else:
            predict = all_pred[idx]

          postive_pred = torch.where(predict >= threshold, 1.0, 0.0)
          postive_pred_index = torch.argwhere(postive_pred)

          if postive_pred_index.shape[0] > 0:
            vid_pred[idx] = 1.0
          else:
            vid_pred[idx] = 0.0

        tp = torch.sum((vid_pred == 1) & (all_labels == 1))
        fp = torch.sum((vid_pred == 1) & (all_labels == 0))
        fn = torch.sum((vid_pred == 0) & (all_labels == 1))

        if tp > 0:
          p = (tp / (tp + fp))
          r = (tp / (tp + fn))
        else:
          continue

        if len(time) > 0:
          time = torch.tensor(time).to(device)
          time = torch.mean(time).to(device)
        else:
          continue

        precision.append(p)
        recall.append(r)
        all_time.append(time)
      all_time = torch.tensor(all_time, dtype=torch.float).to(device)
      precision = torch.tensor(precision, dtype=torch.float).to(device)
      recall = torch.tensor(recall, dtype=torch.float).to(device)
      ap = 0
      if recall.shape[0] == precision.shape[0]:
        for k in range(1, recall.shape[0]):
          h = torch.max(precision[k], precision[k - 1])
          w = recall[k] - recall[k - 1]
          ap += torch.abs(h * w)
      if all_time.shape[0] > 0:
        mtta = torch.mean(all_time)
      else:
        mtta = torch.tensor([0.0])

      r80 = torch.where(recall >= 0.8, recall, 0.0)
      r80_index = torch.argwhere(r80)
      if r80_index.shape[0] > 0:
        tta_r80 = all_time[r80_index]
        tta_r80 = torch.mean(tta_r80)
      else:
        tta_r80 = torch.tensor([0.0])

      if r80_index.shape[0] > 0:
        p_r80 = precision[r80_index]
        p_r80 = torch.mean(p_r80)
      else:
        p_r80 = torch.tensor([0.0])
      if recall.shape[0] > 0:
        recall = torch.mean(recall)
      else:
        recall = torch.tensor([0.0])
      if precision.shape[0] > 0:
        precision = torch.mean(precision)
      else:
        precision = torch.tensor([0.0])



    return ap.item(), mtta.item(), tta_r80.item(), p_r80.item(), precision.item(), recall.item()