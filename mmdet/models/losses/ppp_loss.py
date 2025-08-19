import torch
import torch.nn as nn

from mmdet.registry import MODELS


@MODELS.register_module()
class PPPLoss(nn.Module):
    """Point Process Loss for object detection.

    Args:
        use_sigmoid (bool): Whether to use sigmoid activation.
        loss_weight (float): Weight of the loss.
    """

    def __init__(self, loss_weight=1.0):
        super(PPPLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, inputs, targets, weight=None, avg_factor=1.0):
        """Compute the PPP loss."""
        # H, W = inputs.shape[-2:]

        integral_term = torch.exp(inputs).mean()


        # observation_term = torch.tensor(0.0, device=integral_term.device)
        mask = targets.sum(dim=1)
        
        return (integral_term - (mask * inputs).sum()) / avg_factor