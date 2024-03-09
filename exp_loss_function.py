import torch
import torch.nn as nn

def exp_loss(pred, target, time, toa, device, fps=10.0):

    target_cls = target[:, 1]
    target_cls = target_cls.to(torch.long)
    penalty = -torch.max(torch.zeros_like(toa).to(toa.device, pred.dtype), (toa.to(pred.dtype) - time - 1) / fps).to(device)
    pos_loss = -torch.mul(torch.exp(penalty), - nn.CrossEntropyLoss()(pred, target_cls)).to(device)
    neg_loss = nn.CrossEntropyLoss()(pred, target_cls).to(device)
    loss = torch.mean(torch.add(torch.mul(pos_loss, target[:, 1]), torch.mul(neg_loss, target[:, 0]))).to(device)
    return loss