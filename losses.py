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
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', normalize_by_pos: bool = False):
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
        # If True, normalize loss by number of positive labels instead of all elements
        self.normalize_by_pos = normalize_by_pos

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

        # RetinaNet-style normalization by number of positives (anchors) if requested
        if self.normalize_by_pos:
            # Count positive labels across the batch (multi-label: sum over all classes)
            pos_count = targets.sum()
            # Avoid division by zero when there are no positives in the batch
            return F_loss.sum() / torch.clamp(pos_count, min=1.0)

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
    def __init__(self, samples_per_class, beta=0.9999, gamma=0.5, normalize_by_pos: bool = False):
        """
        Args:
            samples_per_class: List of sample counts [n_D, n_G, n_A]
            beta: Hyperparameter for effective number (0.9999 for very imbalanced)
            gamma: Additional focal parameter (optional)
        """
        super(ClassBalancedLoss, self).__init__()

        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / np.array(effective_num)
        print(f"weights {weights}")

        # Normalize weights
        weights = weights / weights.sum() * len(weights)
        print(f"Normalized weights {weights}")

        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.gamma = gamma
        # If True, normalize loss by number of positive labels instead of all elements
        self.normalize_by_pos = normalize_by_pos

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

        if self.normalize_by_pos:
            pos_count = targets.sum()
            return CB_loss.sum() / torch.clamp(pos_count, min=1.0)

        return CB_loss.mean()


class CombinedLoss(nn.Module):
    """
    Combined Focal Loss and Class Balanced Loss
    """
    def __init__(self, samples_per_class, alpha=None, beta=0.9999, gamma=2.0,
                 focal_weight=0.5, cb_weight=0.5, normalize_by_pos: bool = False):
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
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma, normalize_by_pos=normalize_by_pos)
        # Use pure class-balanced BCE inside the combined loss; focal handled by FocalLoss above
        self.cb_loss = ClassBalancedLoss(samples_per_class, beta=beta, gamma=0, normalize_by_pos=normalize_by_pos)
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
        # Default focal loss with positive-based normalization.
        # If class statistics are provided, use inverse-frequency alpha;
        # otherwise, fall back to the canonical alpha = 0.25.
        if samples_per_class is not None:
            total = sum(samples_per_class)
            alpha = [total / (len(samples_per_class) * n) for n in samples_per_class]
            return FocalLoss(alpha=alpha, gamma=2.0, normalize_by_pos=True)
        # Explicit 0.25 alpha (RetinaNet-style) when no per-class counts are given
        return FocalLoss(alpha=0.25, gamma=2.0, normalize_by_pos=True)

    elif loss_type == 'focal_alpha025':
        # Always use alpha = 0.25 with positive-normalized focal loss
        return FocalLoss(alpha=0.25, gamma=2.0, normalize_by_pos=True)
    
    elif loss_type == 'class_balanced':
        assert samples_per_class is not None, "samples_per_class required for CB loss"
        # Use positive-normalized CB loss by default
        return ClassBalancedLoss(samples_per_class, beta=0.999, gamma=0.0, normalize_by_pos=True)

    elif loss_type == 'combined':
        assert samples_per_class is not None, "samples_per_class required"
        total = sum(samples_per_class)
        alpha = [total / (len(samples_per_class) * n) for n in samples_per_class]
        # Combine focal + CB, both with positive-based normalization
        return CombinedLoss(samples_per_class, alpha=alpha, gamma=2.0, normalize_by_pos=True)

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
