import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import sigmoid_focal_loss as _sigmoid_focal_loss

from ..builder import LOSSES
from .utils import weight_reduce_loss

def equilibrium_loss(pred, label, weight=None, mean_score=None, reduction='mean', avg_factor=None):
    label_one_hot = torch.zeros_like(pred).scatter_(1, label.unsqueeze(1), 1).detach()  # [1024, 1231]

    max_element, _ = pred.max(axis=-1)
    pred = pred - max_element[:, None]  # to prevent overflow

    numerator = mean_score.unsqueeze(0) * torch.exp(pred)
    denominator = numerator.sum(-1, keepdim=True)
    P = numerator / denominator

    probs = (P * label_one_hot).sum(1)  # [1024]
    loss = - probs.log()  # [1024]

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor) # weight: 1024; reduction: 'mean'; avg_factor: float(1024.0)

    return loss


@LOSSES.register_module
class EquilibriumLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 loss_weight=1.0):
        super(EquilibriumLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight

        self.cls_criterion = equilibrium_loss

    def forward(self,
                cls_score,
                label,
                weight=None,
                mean_score=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            mean_score,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_cls
