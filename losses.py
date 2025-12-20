import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    FL(pt) = -a(1-pt)^y log(pt)

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: Weighting factor for each class [a_D, a_G, a_A]
            gamma: Focusing parameter (default: 2.0)
            reduction: 'mean' or 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (B, C) logits
            targets: (B, C) binary labels
        """

        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Calculate pt (predicted probability)
        pt = torch.exp(-BCE_loss)

        F_loss = (1 - pt) ** self.gamma * BCE_loss

        if self.alpha is not None:
            if isinstance(self.alpha, (list, np.ndarray)):
                alpha_t = torch.tensor(self.alpha).to(inputs.device)
                F_loss = alpha_t * F_loss
            else:
                F_loss = self.alpha * F_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


class ClassBalancedLoss(nn.Module):
    """
    Class Balanced Loss based on effective number of samples
    CB = (1 - beta) / (1 - beta^n)

    Reference: Cui et al., "Class-Balanced Loss Based on Effective Number of Samples", CVPR 2019
    """
    def __init__(self, samples_per_class, beta=0.9999, gamma=0.5):
        """
        Args:
            samples_per_class: List of sample counts [n_D, n_G, n_A]
            beta: Hyperparameter for effective number (0.9999 for very imbalanced)
            gamma: Additional focal parameter (optional)
        """
        super(ClassBalancedLoss, self).__init__()

        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / np.array(effective_num)

        # Normalize weights
        weights = weights / weights.sum() * len(weights)

        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.gamma = gamma

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (B, C) logits
            targets: (B, C) binary labels
        """

        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Apply class weights
        weights = self.weights.to(inputs.device)
        CB_loss = weights * BCE_loss

        if self.gamma > 0:
            pt = torch.exp(-BCE_loss)
            CB_loss = ((1 - pt) ** self.gamma) * CB_loss

        return CB_loss.mean()


class CombinedLoss(nn.Module):
    """
    Combined Focal Loss and Class Balanced Loss
    """
    def __init__(self, samples_per_class, alpha=None, beta=0.9999, gamma=2.0, focal_weight=0.5, cb_weight=0.5):
        """
        Args:
            samples_per_class: List of sample counts
            alpha: Alpha for focal loss
            beta: Beta for CB loss
            gamma: Gamma for both losses
            focal_weight: Weight for focal loss in combination
            cb_weight: Weight for CB loss in combination
        """
        super(CombinedLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.cb_loss = ClassBalancedLoss(samples_per_class, beta=beta, gamma=0)
        self.focal_weight = focal_weight
        self.cb_weight = cb_weight

    def forward(self, inputs, targets):
        loss_focal = self.focal_loss(inputs, targets)
        loss_cb = self.cb_loss(inputs, targets)
        return self.focal_weight * loss_focal + self.cb_weight * loss_cb


# Helper function to get loss function
def get_loss_function(loss_type='bce', samples_per_class=None):
    """
    Factory function to get loss function

    Args:
        loss_type: 'bce', 'focal', 'class_balanced', 'combined'
        samples_per_class: [517, 163, 142] for D, G, A
    """
    if loss_type == 'bce':
        return nn.BCEWithLogitsLoss()

    elif loss_type == 'focal':
        # Calculate alpha based on inverse frequency
        if samples_per_class is not None:
            total = sum(samples_per_class)
            alpha = [total / (len(samples_per_class) * n) for n in samples_per_class]
            return FocalLoss(alpha=alpha, gamma=2.0)
        return FocalLoss(gamma=2.0)
    
    elif loss_type == 'class_balanced':
        assert samples_per_class is not None, "samples_per_class required for CB loss"
        return ClassBalancedLoss(samples_per_class, beta=0.999, gamma=0.5)

    elif loss_type == 'combined':
        assert samples_per_class is not None, "samples_per_class required"
        total = sum(samples_per_class)
        alpha = [total / (len(samples_per_class) * n) for n in samples_per_class]
        return CombinedLoss(samples_per_class, alpha=alpha, gamma=2.0)

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")