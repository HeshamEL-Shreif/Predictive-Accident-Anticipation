import torch

def exp_loss(self, pred, target, time, toa, fps=10.0):

    target_cls = target[:, 1]
    target_cls = target_cls.to(torch.long)
    penalty = -torch.max(torch.zeros_like(toa).to(toa.device, pred.dtype), (toa.to(pred.dtype) - time - 1) / fps)
    pos_loss = -torch.mul(torch.exp(penalty), -self.ce_loss(pred, target_cls))
    # negative example
    neg_loss = self.ce_loss(pred, target_cls)

    loss = torch.mean(torch.add(torch.mul(pos_loss, target[:, 1]), torch.mul(neg_loss, target[:, 0])))
    return loss
