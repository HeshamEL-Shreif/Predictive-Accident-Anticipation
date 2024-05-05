import torch
import torch.nn as nn

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def exp_loss(pred, target, time, toa, fps=10.0):
    """
    Description:
      The frame-level loss function on the training dataset:
      The first term in the loss of a positive video, and the second term is the loss of a negative video.
      Negative videos use a regular binary cross entropy loss function.
      But the frame-level loss coefficient for positive videos increases exponentially
      as approaching the accident (e.g., t < ⌧). After that, it reaches the same loss coefficient
      for negative videos. The use of the exponentially increasing loss coefficient for positive videos
      encourages the early anticipation of accidents.
    Parameters:
    - pred (torch.Tensor): Predictions (N x 2).
    - target (torch.Tensor): Labels (N x 2).
    - time (torch.Tensor): Time (N, ).
    - toa (torch.Tensor): Time of accident (N, ).
    - fps (float): Frames per second.
    output:
    loss (torch.Tensor): Loss.
    """
    ce_loss = nn.CrossEntropyLoss()
    # positive example (exp_loss)
    target_cls = target[:, 1]
    target_cls = target_cls.to(torch.long).to(device)
    penalty = -torch.max(torch.zeros_like(toa).to(toa.device, pred.dtype), (toa.to(pred.dtype) - time - 1) / fps)
    pos_loss = -torch.mul(torch.exp(penalty), - ce_loss(pred, target_cls))
    # negative example
    neg_loss = ce_loss(pred, target_cls)

    loss = torch.mean(torch.add(torch.mul(pos_loss, target[:, 1]), torch.mul(neg_loss, target[:, 0])))
    return loss


def Time_to_accident_calc(all_pred, all_labels, time_of_accidents, fps):
    """
    Description:
      The evaluation function for the accident anticipation task.
    Parameters:
    - all_pred (torch.Tensor): Predictions (N x T x  2).
    - all_labels (torch.Tensor): Labels (N x 2).
    - time_of_accidents (torch.Tensor): Time of accidents (N x 1).
    - fps (float): Frames per second.
    output:
    """


    with torch.no_grad():

      time_of_accidents = time_of_accidents.squeeze(1)
      vid_tta = torch.zeros(all_labels.shape[0])
      vid_pred = torch.zeros(all_labels.shape)
      threshold = 0.5

      for idx, toa in enumerate(time_of_accidents):
        if all_labels[idx, 1] == 1:
          t = all_pred[idx, :toa, 1]
          predict = all_pred[idx, :toa, 1]
          t_where = torch.where(t >= threshold, t, 0.0)
          t_index = torch.argwhere(t_where)
          if t_index.shape[0] > 0:
            t_index = t_index[0]
            vid_tta[idx] =  t_index.item()
          else:
            vid_tta[idx] = toa.to(vid_tta.dtype)
        else:
          predict = all_pred[idx, :, 1]

        postive_pred = torch.where(predict >= threshold, predict, 0.0)
        postive_pred_index = torch.argwhere(postive_pred)
        negative_pred = torch.where(predict < threshold, predict, 0.0)
        negative_pred_index = torch.argwhere(negative_pred)

        if postive_pred_index.shape[0] > 0:
          prob = predict[postive_pred_index[0].item()]
          vid_pred[idx] = torch.tensor([1 - prob, prob])

        if negative_pred_index.shape[0] > 0 and postive_pred_index.shape[0] == 0:
          prob = predict[negative_pred_index[0].item()]
          vid_pred[idx] = torch.tensor([1 - prob, prob])

    return vid_tta, vid_pred